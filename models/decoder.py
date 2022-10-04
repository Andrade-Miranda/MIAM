#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:20:46 2021

@author: gustavo
"""
import torch
import torch.nn as nn
from monai.networks.blocks.unetr_block import UnetrUpBlock
from typing import Sequence, Tuple, Union
from .EncoderConvNeXt import LayerNorm
from timm.models.layers import DropPath
from util.block import UPResBlock

#from monai.networks.nets.vit import ViT

class CNN_PuPMLA(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       hidden_size,
       num_modality,
       features,
       norm_name,
       res_block,
       conv_block=True
    ):
        super(CNN_PuPMLA,self).__init__()
        self.decoderList=nn.ModuleList()
        
        for i in range(len(features)):
            if i==0:
                decoder = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_size,# first correspond to the hidden_size coming from transformer
                    out_channels=features[-1] * num_modality,# last feature kernel
                    kernel_size=3,
                    upsample_kernel_size=1,#upsample kernel is 2 always in UNETR
                    norm_name=norm_name,
                    res_block=res_block,
                    )
            else:
                decoder = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=features[-i] * num_modality,
                    out_channels=features[-i-1] * num_modality,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=res_block,
                    )
            self.decoderList.append(decoder)
            
    def forward(self, x):
        y=[]
        for j in range(len(self.decoderList)):
            x = self.decoderList[j](x)
            y.append(x)
        return y  
    

class CNN_PuPMLA_VIT(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       hidden_size,
       num_modality,
       features,
       norm_name,
       res_block,
       conv_block=True
    ):
        super(CNN_PuPMLA_VIT,self).__init__()
        self.decoderList=nn.ModuleList()
        
        for i in range(len(features)):
            if i==0:
                decoder = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=hidden_size,# first correspond to the hidden_size coming from transformer
                    out_channels=features[-1] * num_modality,# last feature kernel
                    kernel_size=3,
                    upsample_kernel_size=2,#upsample kernel is 2 always in UNETR
                    norm_name=norm_name,
                    res_block=res_block,
                    )
            else:
                decoder = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=features[-i] * num_modality,
                    out_channels=features[-i-1] * num_modality,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=res_block,
                    )
            self.decoderList.append(decoder)
            
    def forward(self, x):
        y=[]
        for j in range(len(self.decoderList)):
            x = self.decoderList[j](x)
            y.append(x)
        return y  





###########BASELINE############################################################
class SiameseDecoder(nn.ModuleList):

    def __init__(
            self,
            num_modalities,
            spatial_dims,
            features,
            norm_name,
            kernel_sizes,
            stride
               ):
        self.num_modalities=num_modalities
        super(SiameseDecoder,self).__init__()
        self.decoderList=nn.ModuleList()
        for i in range(len(features)-1):
            decoder= UPResBlock(num_modalities=num_modalities,
                                spatial_dims= spatial_dims,
                                in_channels=features[-i-1],
                                out_channels=features[-i-2],
                                kernel_size=tuple(kernel_sizes[-i-2]),
                                stride=stride[-i-1],
                                norm_name=norm_name
                                      )
            self.decoderList.append(decoder)


    def forward_once(self,x,modal):
        
        x1=x[-1].clone()
        for j in range(len(self.decoderList)):
            x1 = self.decoderList[j](x1,x[-j-2],modal)
            self.y.append(x1)
        return self.y      
    
    def forward(self, input,modal):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        out=[]
        for imag,i in zip(input,modal):
            self.y=[]
            out.append(self.forward_once(imag,i))
        
        return out
###########BASELINE############################################################
    
















class Swim_Decoder(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       num_modality,
       features,
       norm_name,
       res_block,
       conv_block=True
    ):
        super(Swim_Decoder,self).__init__()
        self.decoderList=nn.ModuleList()
        
        for i in range(len(features)):
            if len(features)==i+1:
                decoder = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=features[0] * num_modality,
                        out_channels=features[0] * num_modality,
                        kernel_size=3,
                        upsample_kernel_size=4,
                        norm_name=norm_name,
                        res_block=res_block,
                        )
            elif i==0:
                decoder = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=features[-i-1] * num_modality,
                        out_channels=features[-i-2] * num_modality,
                        kernel_size=3,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        res_block=res_block,
                        )
            else:
                decoder = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=features[-i-1] * num_modality,
                        out_channels=features[-i-2] * num_modality,
                        kernel_size=3,
                        upsample_kernel_size=(1,2,2),
                        norm_name=norm_name,
                        res_block=res_block,
                        )
            self.decoderList.append(decoder)
            
    def forward(self, x):
        y=[]
        for j in range(len(self.decoderList)):
            x = self.decoderList[j](x)
            y.append(x)
        return y  





class ConVneXtDecoder(nn.ModuleList):

    def __init__(
       self,
       features,
    ):
        super(ConVneXtDecoder,self).__init__()
        self.decoderList=nn.ModuleList()
        
        for i in range(len(features)):
            if len(features)==i+1:
                decoder = ConVneXtUpBlock(
                        in_channels=features[0],# first correspond to the hidden_size coming from transformer
                        out_channels=features[0]//2,# last feature kernel
                        kernel_size=3
                        )
            else:
                decoder = ConVneXtUpBlock(
                        in_channels=features[-i-1],# first correspond to the hidden_size coming from transformer
                        out_channels=features[-i-2],# last feature kernel
                        kernel_size=3
                        )
            self.decoderList.append(decoder)
            
    def forward(self, x):
        y=[]
        for j in range(len(self.decoderList)):
            x = self.decoderList[j](x)
            y.append(x)
        return y  


class ConVneXtUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """
        drop_path=0.
        layer_scale_init_value=1e-6
        super().__init__()
        
        self.transp_conv=nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dwconv = nn.Conv3d(out_channels*2, out_channels, kernel_size=3, padding=1, groups=out_channels) # depthwise conv
        self.norm = LayerNorm(out_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(out_channels, out_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(out_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        x = self.transp_conv(inp)
        input = x
        x = torch.cat((x, skip), dim=1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3,4, 1) # (N, C, D, H, W) -> (N, D,H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2,3) # (N, D,H, W, C) -> (N, C, D,H, W)

        x = input + self.drop_path(x)
        return x



































    
class SwimBS_Decoder(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       num_modality,
       features,
       norm_name,
       res_block,
       conv_block=True
    ):
        super(SwimBS_Decoder,self).__init__()
        self.decoderList=nn.ModuleList()
        
        for i in range(len(features)):
            if len(features)==i+1:
                decoder = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=features[0] * num_modality,
                        out_channels=features[0] * num_modality,
                        kernel_size=3,
                        upsample_kernel_size=(2,4,4),
                        norm_name=norm_name,
                        res_block=res_block,
                        )
            elif i==0:
                decoder = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=features[-i-1] * num_modality,
                        out_channels=features[-i-2] * num_modality,
                        kernel_size=3,
                        upsample_kernel_size=2,
                        norm_name=norm_name,
                        res_block=res_block,
                        )
            else:
                decoder = UnetrUpBlock(
                        spatial_dims=spatial_dims,
                        in_channels=features[-i-1] * num_modality,
                        out_channels=features[-i-2] * num_modality,
                        kernel_size=3,
                        upsample_kernel_size=(1,2,2),
                        norm_name=norm_name,
                        res_block=res_block,
                        )
            self.decoderList.append(decoder)
            
    def forward(self, x):
        y=[]
        for j in range(len(self.decoderList)):
            x = self.decoderList[j](x)
            y.append(x)
        return y  