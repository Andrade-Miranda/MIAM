#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:26:58 2021

@author: gustavo
"""



import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer,get_act_layer,get_norm_layer
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
from .decoder import CNN_PuPMLA
from .encoder import SiameseEncoder




class SharedCNN_VITNaive(nn.Module):

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
        filters_Encoder=opt.filters_Encoder
        res_block=opt.res_block
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        self.opt=opt
        
        self.numModal= opt.input_nc
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
        super(SharedCNN_VITNaive,self).__init__()
        
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        
        """ -------Shared encoder with specialize normalization---------------"""
        self.encodModalities = SiameseEncoder(
            num_modalities=self.numModal,
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            kernel_sizes=opt.conv_kernel_sizes,
            stride=opt.pool_op_kernel_sizes
            )
        self.MaxPool=nn.MaxPool3d(3, stride=2,padding=0,dilation=1,ceil_mode=True)
        """ ----------------------------------------------------------------"""      
       
        
        """ -------------------VIT encoders------------------------------- """      
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = num_layers
        downfactor=int(2**(len(filters_Encoder)))
        img_size = tuple([int((x/downfactor)) for x in img_size])
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT(
            in_channels=filters_Encoder[-1]*in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        """ -------------------CNN decoders------------------------------- """
        self.UpsamplingConv=get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            stride=2,
            conv_only=True,
            is_transposed=True,
            )
             
        self.decoder=CNN_PuPMLA(
                   spatial_dims=spatial_dims,
                   hidden_size=hidden_size,
                   num_modality=in_channels,
                   features=filters_Encoder,
                   norm_name=norm_name,
                   res_block=res_block,            
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0]* in_channels, out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        
        """ -------------------CNN reshape when last layer doesnt match------------------------------- """        
        if self.patch_size[0] > 1: 
            self.reshapeConv=get_conv_layer(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=3,
                stride=self.patch_size,
                conv_only=True,
                is_transposed=True,
                )
        """ -------------------------------------------------- """             
        
        self.numModal=in_channels #gloabal variable to keep the number of modalities
        self.numConvLevel=len(filters_Encoder)

        
    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    

    def forward(self, x_in):
        
        modal= [i for i in range(self.numModal)]
        data= [x_in[:,i,None,:] for i in range(self.numModal)]
        encModal=self.encodModalities(data,modal=modal)#modify for multiples modalities
    
        skip_connections=[]
        for i in range(self.numConvLevel):
            skip=[]
            for j in range(self.numModal):
                skip.append(encModal[j][i])# list of tensor
            skip_connections.append(torch.cat(skip,1))
            
        maxPool=self.MaxPool(skip_connections[-1])    
    
        outViT, hidden_states_out = self.vit(maxPool)
        decfinal = self.proj_feat(outViT, self.hidden_size, self.feat_size)
        decfinal= self.UpsamplingConv(decfinal)
        
        j=-1
        for numdec in range(self.numConvLevel):
            if numdec==0 and skip_connections[j].shape[-1]!=decfinal.shape[-1]:
                decfinal=self.reshapeConv(decfinal)
            decfinal = self.decoder.decoderList[numdec](decfinal,skip_connections[j])#change to only concat
            j=j-1
        
        return self.out(decfinal)
        

    def name(self):
        return 'Shared Encoder with instance normalization per modality'


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







































