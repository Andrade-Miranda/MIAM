#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:26:58 2021

@author: gustavo
"""



import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from timm.models.layers import trunc_normal_
from torch.nn import init
from util.util import print_network
from .decoder import ConVneXtDecoder
import einops
from .EncoderConvNeXt import convnext_tiny,convnext_small,convnext_base,convnext_large,convnext_xlarge 
from timm.models.registry import register_model

""" Basic Unet 
    A UNet Encoder block  implementation with 1D/2D/3D supports.
        Based on:
Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
Morphometry". Nature Methods 16, 67–70 (2019), DOI:http://dx.doi.org/10.1038/s41592-018-0261-2
    Adapted from Monai
"""
""" CNN heavy --CNN_h
"""
class ConvNeXt_Unet(nn.Module):

    def __init__(self, opt):
        
        in_channels=opt.input_nc
        out_channels= opt.output_nc
        img_size=opt.imageSize
        filters_Encoder=opt.filters_Encoder
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        name_model='convnext_small'
        self.opt=opt
        
        self.Multiples_encoder=False
        self.numModal=in_channels #gloabal variable to keep the number of modalities
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
        super(ConvNeXt_Unet,self).__init__()
        
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        
        """ -------multipath encoders------------------------------------- """
        self.encodModalities,filters_Encoder =ConvNextEncoders(in_channels,name_model,Multiples_encoder=self.Multiples_encoder)
        self.conInp = nn.Conv3d(in_channels, filters_Encoder[0]//2, kernel_size=3, padding=1, groups=out_channels) # depthwise conv
        self.numConvLevel=len(filters_Encoder)
        """ ----------------------------------------------------------------"""      
       
        self.decoder=ConVneXtDecoder(
                    features=filters_Encoder,
                    )
        
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0]//2, out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        


    def forward(self, x_in):
        
        if self.Multiples_encoder:
            encModal=[]
            numnoda=0
            for modality in self.encodModalities:
                encModal.append(modality(x_in[:,numnoda:numnoda+1]))# 1 modalidad
                numnoda=numnoda+1
        else:
            encModal,skip=self.encodModalities[0](x_in)
        
        # skip_connections=[]
        # for i in range(self.numConvLevel):
        #     skip=[]
        #     for j in range(self.numModal):
        #         skip.append(encModal[j][i])
        #     skip_connections.append(torch.cat(skip,1))
        j=-2
        for numdec in range(self.numConvLevel):
            if numdec==self.numConvLevel-1:
               encModal = self.decoder.decoderList[numdec](encModal,self.conInp(skip[j]))#change to only concat
            else:
                encModal = self.decoder.decoderList[numdec](encModal,skip[j])#change to only concat
            j=j-1
        
        
        return self.out(encModal)
    

    def name(self):
        return 'CoNVNext'


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
                    #init.normal_(m.weight.data, 0.0, 0.02)
                    trunc_normal_(m.weight, std=.02)
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
                init.constant_(m.bias, 0.0)
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


@register_model
def ConvNextEncoders(in_channels,name_model,Multiples_encoder=False,**kwargs):
        if Multiples_encoder:
            in_chans=1
        else:
            in_chans=in_channels
            
        encoderList=nn.ModuleList()        
        if name_model=='convnext_tiny':
            encoder=convnext_tiny(in_chans=in_chans)
            features=[96,192,384,768]
        elif name_model=='convnext_small':
            encoder=convnext_small(in_chans=in_chans)
            features=[96,192,384,768]
        elif name_model=='convnext_base':
            encoder=convnext_base(in_chans=in_chans)
            features=[128,256,512,1024]
        elif name_model=='convnext_large':
            encoder=convnext_large(in_chans=in_chans)
            features=[192,384,768,1536]
        elif name_model=='convnext_xlarge':
            encoder=convnext_xlarge(in_chans=in_chans)
            features=[256,512,1024,2048]
        else:
            raise NameError('no encoder found')
        
        if Multiples_encoder:
            for i in range(len(in_channels)):
                    encoderList.append(encoder)
        else:
            encoderList.append(encoder)
        
        return encoderList,features
            










































