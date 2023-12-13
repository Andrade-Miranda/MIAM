#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:26:58 2021

@author: gustavo
"""


import math

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from .ViT_StreamSM import ViT_S
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
from util.block import FusedGatedUnit
from .encoder import BasicUnetEnc
from .decoder import MCNN_VIT_decoder
import einops

  

class MultiCNNHeavy_VITsingle(nn.Module):

    def __init__(self, opt):
        
        in_channels=opt.input_nc
        out_channels= opt.output_nc
        img_size=opt.imageSize
        feature_size=opt.patchSize
        hidden_size=opt.hidden_size
        mlp_dim= opt.mlp_dim
        num_heads=opt.num_heads
        num_layers=opt.num_layers
        pos_embed=opt.pos_embed
        norm_name=opt.norm_name
        filters_Encoder=opt.filters_Encoder[:-1]
        res_block=opt.res_block
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        kernel_sizes=opt.conv_kernel_sizes[:-1]
        stride=opt.pool_op_kernel_sizes[:-1]
        self.opt=opt
        
        self.numModal=in_channels #gloabal variable to keep the number of modalities
        self.numConvLevel=len(filters_Encoder)
        """
        Args:
            in_channels: dimension of input channels (Modalities).
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        """
        super(MultiCNNHeavy_VITsingle,self).__init__()
        

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        
        """ -------multipath encoders------------------------------------- """
        
        self.encodModalities = nn.ModuleList()
        for i in range(in_channels):
            self.encodModalities.append(BasicUnetEnc(
            spatial_dims= spatial_dims,
            in_channels= 1,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            kernel_sizes=kernel_sizes,
            stride=stride,
            )
                )
        """ ----------------------------------------------------------------"""      
       
        
        """ -------------------VIT encoders------------------------------- """      
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        a,b,c=0,0,0
        for i,j,k in self.opt.pool_op_kernel_sizes[:-1]:
            if i==2: a+=1 
            if j==2: b+=1 
            if k==2: c+=1
        self.opt.num_pool_per_axis=[a,b,c]

        self.num_layers = num_layers
        downfactor=[int(2**(i)) for i in self.opt.num_pool_per_axis] #for extra maxpooling
        img_size = tuple([math.ceil((x/downfactor[i])) for i,x in enumerate(img_size)])
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT_S(
            numModal=self.numModal,
            in_channels=filters_Encoder[-1],
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            pos_embed=pos_embed,
            num_layers=self.num_layers,
            num_heads=num_heads,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            fusion=self.opt.Earlyfusion
        )
        """ ------------------------------------------------------------- """  
        self.ProjShared=FusedGatedUnit(hidden_size,
               hidden_size)
        
        
        """ -------------------CNN decoders------------------------------- """
        from copy import deepcopy
        filters_EncVit=list(deepcopy(filters_Encoder))[:-1]
        filters_EncVit.append(hidden_size)
        self.decoder=MCNN_VIT_decoder(
                   spatial_dims=spatial_dims,
                   num_modality=in_channels,
                   features=tuple(filters_EncVit),
                   norm_name=norm_name,
                   res_block=res_block,
                   kernel_sizes=kernel_sizes,
                   stride=stride,             
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0] * in_channels, out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        
    
        """ -------------------CNN reshape when last layer doesnt match------------------------------- """        
        if self.patch_size[0] > 1: 
            self.reshapeConv=get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=hidden_size*self.numModal,
                out_channels=hidden_size*self.numModal,
                kernel_size=3,
                stride=self.patch_size,
                conv_only=True,
                is_transposed=True,
                )
        """ -------------------------------------------------- """             
        
    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    

    def forward(self, x_in):
        
        encModal=[]
        numnoda=0
        for modality in self.encodModalities:
            encModal.append(modality(x_in[:,numnoda:numnoda+1]))# 1 modalidad
            numnoda=numnoda+1
        
        skip_connections=[]
        for i in range(self.numConvLevel):
            skip=[]
            for j in range(self.numModal):
                skip.append(encModal[j][i])
            skip_connections.append(torch.cat(skip,1))
        
        outViT, _ = self.vit(skip)
        outViT = einops.rearrange(outViT, "b (Np n) H -> b n Np H",n=numnoda)
        outViT=self.ProjShared(outViT)
        outViT=self.proj_feat(outViT, self.hidden_size, self.feat_size)

        output=self.decoder(outViT,skip_connections)
        
        return self.out(output[-1])
    

    def name(self):
        return 'Multipath with Heavy CNN + VIT Single'


        """--------------------Initialize network weights.---------------"""   
    def init_weights(self,net, init_type='normal', init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, 0.02)
                    #m.weight.data.normal_(0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
        
    def init_net(self,model, init_type='normal', init_gain=0.02):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

        Return an initialized network.
        """
        if not self.opt.gpu_ids:
            model = model.to(self.opt.device)
        elif self.opt.gpu_ids[0]>1:
            assert(torch.cuda.is_available())
            model = torch.nn.DataParallel(model, list(range(self.opt.gpu_ids[0]))).to(self.opt.device)  # multi-GPUs
        else:
            model = model.to(self.opt.device)
        print_network(model)
        """---------------------"""
        self.init_weights(model, init_type, init_gain=init_gain)
        return model
    """--------------------------------------------------------------------""" 













































