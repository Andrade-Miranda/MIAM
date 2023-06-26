#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:51:52 2022

@author: gustavo
"""
from monai.networks.blocks.unetr_block import UnetrBasicBlock,UnetrUpBlock
from util.block import UnetResBlock,UnetResBlockNormSE
import torch.nn as nn
import torch

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
       spatial_dims,
       in_channels,
       features,
       norm_name,
       res_block,
       kernel_sizes,
       stride
    ):
        super(BasicUnetEnc,self).__init__()
        self.encoderList=nn.ModuleList()
        features=tuple([in_channels])+features

        for i in range(len(features)-1):             
            encoder= UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=features[i],
                    out_channels=features[i+1],
                    kernel_size=tuple(kernel_sizes[i]),
                    stride=tuple(stride[i]),
                    norm_name=norm_name,
                    res_block=res_block
                    )
            self.encoderList.append(encoder)


    def forward(self, x):
        y=[]
        for j in range(len(self.encoderList)):
            x = self.encoderList[j](x)
            y.append(x)
        return y 




if __name__ == "__main__":
    encoder= BasicUnetEnc(spatial_dims=3,
                            in_channels=1,
                            features=(32, 64, 128, 256, 320,512, 768),
                            norm_name='instance',
                            res_block=True,
                            kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                            stride=[[1,1,1],[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
                            )
    x=torch.rand((1,1,20,384,384))
    z=encoder(x)

    #decoder= CNN_PuPMLA(spatial_dims=3,
    #                        num_modality=1,
    #                        features=(32, 64, 128, 256,320, 512, 768),
    #                        norm_name='instance',
    #                        res_block=True,
    #                        kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3]],
    #                        stride=[[1,1,1],[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2],[1,2,2]]
    #                       )
    #y=decoder(z)



