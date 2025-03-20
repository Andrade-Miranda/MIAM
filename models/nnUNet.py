#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:28:54 2021

@author: gustavo
"""


import math
import torch
import torch.nn as nn
import numpy as np

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer
#from monai.networks.blocks.unetr_block import UnetrBasicBlock
#from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
#from monai.networks.blocks.unetr_block import UnetrUpBlock
from .encoder import BasicUnetEnc
from .decoder import CNN_decoder
from models.dynamic_network_architectures.architectures.unet import PlainConvUNet

class nnUNetPlain(nn.Module):
    
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
        deep_supervision= False
        nonlin_first= False
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        
        self.opt=opt
        
        super(nnUNetPlain,self).__init__()
        
        self.model = PlainConvUNet(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,n_conv_per_stage, num_classes,
                                n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,nonlin_kwargs, 
                                deep_supervision=deep_supervision)
        
        
    def forward(self, x_in):
        return self.model(x_in)
        
    def name(self):
        return 'nnUNet Plain'

    def init_net(self,model, init_type='kaiming', init_gain=0.02):
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
        print('#model created')
        """---------------------"""
        self.init_weights(model, init_type, init_gain=init_gain)
        return model
    """--------------------------------------------------------------------""" 

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
        



class nnUNetvResnet(nn.Module):

    def __init__(self, opt):
        
        in_channels=opt.input_nc
        out_channels= opt.output_nc
        norm_name=opt.norm_name
        filters_Encoder=opt.filters_Encoder
        res_block=opt.res_block
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        kernel_sizes=opt.conv_kernel_sizes
        stride=opt.pool_op_kernel_sizes
        self.opt=opt
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
        super(nnUNetvResnet,self).__init__()
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        """ ------- encoders------------------------------------- """
        self.encodModalities = BasicUnetEnc(
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            kernel_sizes=kernel_sizes,
            stride=stride,
            )
        """ ----------------------------------------------------------------"""              

        """ -------------------CNN decoders------------------------------- """             
        self.decoder=CNN_decoder(
                   spatial_dims=spatial_dims,
                   multipath=1,
                   features=filters_Encoder,
                   norm_name=norm_name,
                   res_block=res_block,
                   kernel_sizes=kernel_sizes,
                   stride=stride,
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0], out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        

        self.numModal=in_channels #gloabal variable to keep the number of modalities
        self.numConvLevel=len(filters_Encoder)

        
    def forward(self, x_in):
        
        encModal=self.encodModalities(x_in)
            
        output=self.decoder(encModal)

        return self.out(output[-1])
    

    def name(self):
        return 'Custom nnUNeT'


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
        print('#model created')
        """---------------------"""
        #self.init_weights(model, init_type, init_gain=init_gain)
        return model
    """--------------------------------------------------------------------""" 
































