#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:28:54 2021

@author: gustavo
"""


import math
import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
from models.dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from models.dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from models.dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

  
class CNNHeavy_VITNaive(nn.Module):

    def __init__(self, opt):
        
        input_channels=opt.input_nc
        n_stages=len(opt.filters_Encoder)
        features_per_stage=opt.filters_Encoder
        conv_op=nn.Conv3d
        kernel_sizes=opt.conv_kernel_sizes
        strides=opt.pool_op_kernel_sizes
        n_conv_per_stage= tuple(np.repeat(2,len(opt.filters_Encoder)))
        num_classes=opt.output_nc
        n_conv_per_stage_decoder= tuple(np.repeat(2,len(opt.filters_Encoder[1:])))
        conv_bias= False
        norm_op=  nn.InstanceNorm3d if opt.norm_name == 'instance' else nn.BatchNorm3d
        norm_op_kwargs= {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout3d
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        nonlin= nn.LeakyReLU
        nonlin_kwargs= {'negative_slope': 1e-2, 'inplace': True}
        deep_supervision= opt.DeepSupervision
        nonlin_first= False
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        #transformer
        feature_size=opt.patchSize
        hidden_size=opt.hidden_size
        mlp_dim= opt.mlp_dim
        num_heads=opt.num_heads
        num_layers=opt.num_layers
        pos_embed=opt.pos_embed
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        
        self.opt=opt

        super(CNNHeavy_VITNaive,self).__init__()
        
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        ######nnUnet ecoder###################################
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        

        """ -------------------VIT encoders------------------------------- """
        a,b,c=0,0,0
        for i,j,k in self.opt.pool_op_kernel_sizes[:-1]:
            if i==2: a+=1 
            if j==2: b+=1 
            if k==2: c+=1
        self.opt.num_pool_per_axis=[a,b,c]

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = num_layers
        downfactor=[int(2**(i)) for i in self.opt.num_pool_per_axis] #for extra maxpooling
        img_size = tuple([math.ceil((x/downfactor[i])) for i,x in enumerate(img_size)])
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT(
            in_channels=features_per_stage[-1],
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
        """ ------------------------------------------------------------- """      

        ######nnUnet decoder###################################
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first) 
              
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

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    

    def forward(self,x_in,DeppSuper):
        
        encModal=self.encodModalities(x_in)
        
        outViT, _ = self.vit(encModal[-1])
        outViT = self.proj_feat(outViT, self.hidden_size, self.feat_size)
        output=self.decoder(outViT,encModal)
        
        if DeppSuper:
            return self.deepSupervision(output)
        else:
            return self.out(output[-1])
        
    

    def name(self):
        return 'Multipath with Heavy CNN + VIT Naive'


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
            if classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
        return net
        
    def init_net(self,model, init_type='normal', init_gain=0.02):
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
            gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

        Return an initialized network.
        """
        if self.opt.pretrained: # to check the keys in the dictionary
            if isinstance(self.opt.pretrained, str):
                model.load_state_dict(torch.load(self.opt.pretrained,map_location=self.opt.device),strict=False)
                print('initialize network with pretained weights %s' % self.opt.pretrained)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            model=self.init_weights(model, init_type, init_gain=init_gain)
        
        model=self.wrap_model(model)
        """---------------------"""
        print_network(model)
        print('#model created')
        """---------------------"""
        return model

    """--------------------------------------------------------------------""" 

    def wrap_model(self,model):
        """
        1. Distribute model or not
        2. Rewriting batch size and workers
        """
        args = self.opt
        assert model is not None, "Please build model before wrapping model"
        
        if args.distributed:
            ngpus_per_node = args.ngpus_per_node
            # Apply SyncBN
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.batch_size = args.batch_size // ngpus_per_node
                self.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
                print("=> Finish adapting batch size and workers according to gpu number")
                model = nn.parallel.DistributedDataParallel(model, 
                                                            device_ids=[args.gpu],
                                                            find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        else :
            model = model.to(self.opt.device)
        
        return model   














































    
 
