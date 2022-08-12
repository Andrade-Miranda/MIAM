
"""
Created on Wed Mar  9 12:01:48 2022

@author: gustavo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:28:54 2021

@author: gustavo
"""


import math
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer
from .ViT_StreamSM import ViT_M
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
from .decoder import CNN_PuPMLA


from typing import Optional, Sequence, Tuple, Union
import numpy as np
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import optional_import
einops, _ = optional_import("einops")


class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        num_modalities: int,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.num_modalities=num_modalities
        self.conv1 = get_conv_layer(
            spatial_dims,in_channels,out_channels,kernel_size=kernel_size,stride=stride,dropout=dropout,conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.conv3 = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        
        self.norm1 = nn.ModuleList(
            [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels) for i in range(num_modalities)]
        )
        
        self.norm2 = nn.ModuleList(
            [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels) for i in range(num_modalities)]
        )
        
        self.norm3 = nn.ModuleList(
            [get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels) for i in range(num_modalities)]
        )        
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

    def forward(self, inp,j):
        residual = inp
        if j==0:
            x = einops.rearrange(inp, "b m d w l -> (b m) 1 d w l")
        else:
            x = inp
            
        out = self.conv1(x)
        out = einops.rearrange(out, " (b m) f d w l -> b m f d w l", m=self.num_modalities)
        for i in range(out.shape[1]):
            out[:,i,:,:,:,:] = self.norm1[i](out[:,i,:,:,:,:])
        out = self.lrelu(out)
        out = einops.rearrange(out, "b m f d w l -> (b m) f d w l")
        out = self.conv2(out)
        out = einops.rearrange(out, " (b m) f d w l -> b m f d w l", m=self.num_modalities)
        for i in range(out.shape[1]):
            out[:,i,:,:,:,:] = self.norm2[i](out[:,i,:,:,:,:])        
        if self.downsample:
            if j==0:
                residual = einops.rearrange(residual, "b m d w l -> (b m) 1 d w l")
            residual = self.conv3(residual)
            residual = einops.rearrange(residual, " (b m) f d w l -> b m f d w l", m=self.num_modalities)
            for i in range(residual.shape[1]):
                residual[:,i,:,:,:,:] = self.norm3[i](residual[:,i,:,:,:,:])
        out += residual
        out = self.lrelu(out)
        out = einops.rearrange(out, " b m f d w l -> (b m) f d w l")
        return out


""" Basic Unet 
    A UNet Encoder block  implementation with 1D/2D/3D supports.
        Based on:
Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
Morphometry". Nature Methods 16, 67–70 (2019), DOI:http://dx.doi.org/10.1038/s41592-018-0261-2
    Adapted from Monai
"""
""" CNN heavy --CNN_h
"""
class BasicUnetEnc(nn.ModuleList):

    def __init__(
       self,
       num_modalities,
       spatial_dims,
       in_channels,
       features,
       norm_name
    ):
        self.num_modalities=num_modalities
        super(BasicUnetEnc,self).__init__()
        self.encoderList=nn.ModuleList()
        for i in range(len(features)):
            if i==0:
                encoder= UnetResBlock(num_modalities=num_modalities,
                                      spatial_dims= spatial_dims,
                                      in_channels=in_channels,
                                      out_channels=features[i],
                                      kernel_size=3,
                                      stride=1,
                                      norm_name=norm_name
                                      )
            else:
                encoder= UnetResBlock(num_modalities=num_modalities,
                    spatial_dims=spatial_dims,
                    in_channels=features[i-1],
                    out_channels=features[i],
                    kernel_size=3,
                    stride=2,
                    norm_name=norm_name
                    )
            self.encoderList.append(encoder)


    def forward(self, x):
        y=[]
        for j in range(len(self.encoderList)):
            x = self.encoderList[j](x,j)
            y.append(einops.rearrange(x, " (b m) f d w l -> b (f m) d w l",m=self.num_modalities))
        return y       




class SharedCNN_VITMultiple(nn.Module):

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
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        res_block=True
        self.opt=opt
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
        super(SharedCNN_VITMultiple,self).__init__()
        
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        self.features=filters_Encoder
        
        """ -------multipath encoders------------------------------------- """
        
        self.encodModalities = BasicUnetEnc(
            num_modalities=in_channels,
            spatial_dims= spatial_dims,
            in_channels= 1,
            features= filters_Encoder,
            norm_name=norm_name,
            )
        """ ----------------------------------------------------------------"""      
      
        self.MaxPool=nn.MaxPool3d(3, stride=2,padding=0,dilation=1,ceil_mode=True)
        
        """ -------------------VIT encoders------------------------------- """      
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = num_layers
        downfactor=int(2**(len(filters_Encoder))) #for extra maxpooling
        img_size = tuple([math.ceil((x/downfactor)) for x in img_size])
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT_M(
            numModal=self.numModal,
            in_channels=filters_Encoder[-1],
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
        
        
        """ -------------------CNN decoders------------------------------- """
        self.UpsamplingConv=get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=hidden_size*self.numModal,
            out_channels=hidden_size*self.numModal,
            kernel_size=3,
            stride=2,
            conv_only=True,
            is_transposed=True,
            )
             
        self.decoder=CNN_PuPMLA(
                   spatial_dims=spatial_dims,
                   hidden_size=hidden_size*self.numModal,
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
        new_view = (x.size(0), *feat_size, hidden_size*self.numModal)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    

    def forward(self, x_in):
        
        encModal=self.encodModalities(x_in)
        maxPool=self.MaxPool(encModal[-1])
        maxPool=list(torch.split(maxPool,self.features[-1],dim=1))
        outViT, hidden_states_out = self.vit(maxPool)
        decfinal = self.proj_feat(outViT, self.hidden_size, self.feat_size)
        decfinal= self.UpsamplingConv(decfinal)
        
        j=-1
        for numdec in range(self.numConvLevel):
            if numdec==0 and encModal[j].shape[-1]!=decfinal.shape[-1]:
                decfinal=self.reshapeConv(decfinal)
            decfinal = self.decoder.decoderList[numdec](decfinal,encModal[j])#change to only concat
            j=j-1
        
        return self.out(decfinal)
    

    def name(self):
        return 'shared Heavy CNN + VIT Multiple'


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

