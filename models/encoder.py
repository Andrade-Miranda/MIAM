#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:51:52 2022

@author: gustavo
"""
from util.block import UnetResBlock,UnetResBlockNormSE
import torch.nn as nn

###########BASELINE############################################################
class SiameseEncoder(nn.ModuleList):

    def __init__(
            self,
               num_modalities,
               spatial_dims,
               in_channels,
               features,
               norm_name,
               kernel_sizes,
               stride
               ):
        self.num_modalities=num_modalities
        super(SiameseEncoder,self).__init__()
        self.encoderList=nn.ModuleList()
        for i in range(len(features)):
            if i==0:
                encoder= UnetResBlock(num_modalities=num_modalities,
                                      spatial_dims= spatial_dims,
                                      in_channels=1,
                                      out_channels=features[i],
                                      kernel_size=tuple(kernel_sizes[i]),
                                      stride=1,
                                      norm_name=norm_name
                                      )
            else:
                encoder= UnetResBlock(num_modalities=num_modalities,
                    spatial_dims=spatial_dims,
                    in_channels=features[i-1],
                    out_channels=features[i],
                    kernel_size=tuple(kernel_sizes[i]),
                    stride=tuple(stride[i-1]),
                    norm_name=norm_name
                    )
            self.encoderList.append(encoder)


    def forward_once(self, x,modal):
        y=[]
        for j in range(len(self.encoderList)):
            x = self.encoderList[j](x,modal)
            y.append(x)
        return y       
    
    def forward(self, input,modal):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        z=[]
        for imag,i in zip(input,modal):
            z.append(self.forward_once(imag,i))
        
        return z
###########BASELINE############################################################


###########BASELINE+NORMSE############################################################
class SiameseEncoderNormSE(nn.ModuleList):

    def __init__(
            self,
               num_modalities,
               spatial_dims,
               in_channels,
               features,
               norm_name,
               kernel_sizes,
               stride
               ):
        self.num_modalities=num_modalities
        super(SiameseEncoderNormSE,self).__init__()
        self.encoderList=nn.ModuleList()
        for i in range(len(features)):
            if i==0:
                encoder= UnetResBlockNormSE(num_modalities=num_modalities,
                                      spatial_dims= spatial_dims,
                                      in_channels=1,
                                      out_channels=features[i],
                                      kernel_size=tuple(kernel_sizes[i]),
                                      stride=1,
                                      norm_name=norm_name
                                      )
            else:
                encoder= UnetResBlockNormSE(num_modalities=num_modalities,
                    spatial_dims=spatial_dims,
                    in_channels=features[i-1],
                    out_channels=features[i],
                    kernel_size=tuple(kernel_sizes[i]),
                    stride=tuple(stride[i-1]),
                    norm_name=norm_name
                    )
            self.encoderList.append(encoder)


    def forward_once(self, x,modal):
        y=[]
        for j in range(len(self.encoderList)):
            x = self.encoderList[j](x,modal)
            y.append(x)
        return y       
    
    def forward(self, input,modal):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        z=[]
        for imag,i in zip(input,modal):
            z.append(self.forward_once(imag,i))
        
        return z
###########BASELINE############################################################