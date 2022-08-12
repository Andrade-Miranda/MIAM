#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:16:09 2021

@author: gustavo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:28:54 2021

@author: gustavo
"""



import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep


""" Basic Unet 
    A UNet Encoder block  implementation with 1D/2D/3D supports.
        Based on:
Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
Morphometry". Nature Methods 16, 67–70 (2019), DOI:http://dx.doi.org/10.1038/s41592-018-0261-2
    Adapted from Monai
"""
class BasicUnetEnc(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       in_channels,
       features,
       norm_name,
       res_block,
       conv_block=True
    ):
        super(BasicUnetEnc,self).__init__()
        self.encoderList=nn.ModuleList()
        self.encoders=[]
        
        for i in range(len(features)):
            if i==0:
                encoder= UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=1,
                    out_channels=features[i],
                    kernel_size=3,
                    stride=1,
                    norm_name=norm_name,
                    res_block=res_block,
                    )
            else:
                encoder= UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=features[i-1],
                    out_channels=features[i],
                    kernel_size=3,
                    stride=2,
                    norm_name=norm_name,
                    res_block=res_block,
                    )
            self.encoderList.append(encoder)


    def forward(self, x):
        y=[]
        for j in range(len(self.encoderList)):
            x = self.encoderList[j](x)
            y.append(x)
        return y       






class Hierarchical2TModel(nn.Module):

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
        """
        Args:
            in_channels: dimension of input channels.
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
        super(Hierarchical2TModel,self).__init__()
        
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
 
        self.encodModalities1=BasicUnetEnc(
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            )
        self.encodModalities2=BasicUnetEnc(
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            )
        self.encodModalities3=BasicUnetEnc(
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            )
        self.encodModalities4=BasicUnetEnc(
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            )
        
        
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = num_layers
        downfactor=int(2**(len(filters_Encoder)-1))
        img_size = tuple([int((x/downfactor)) for x in img_size])
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit1 = ViT(
            in_channels=filters_Encoder[-1]*2,
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
        
        self.vit2 = ViT(
            in_channels=filters_Encoder[-1]*2,
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
        
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=filters_Encoder[-1] * 4,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=filters_Encoder[-1] * 4,
            out_channels=filters_Encoder[-2] * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=filters_Encoder[-2] * 4,
            out_channels=filters_Encoder[-3] * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=filters_Encoder[-3] * 4,
            out_channels=filters_Encoder[0] * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
  
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0] * 4, out_channels=out_channels)

        
    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x




    def forward(self, x_in):
        encModal1=self.encodModalities1(x_in[:,0:1])
        encModal2=self.encodModalities2(x_in[:,1:2])
        encModal3=self.encodModalities3(x_in[:,2:3])
        encModal4=self.encodModalities4(x_in[:,3:4])
        
        enc4_All=torch.cat((encModal1[3],encModal2[3],encModal3[3],encModal4[3]),1)# to improve since is not automatic
        enc3_All=torch.cat((encModal1[2],encModal2[2],encModal3[2],encModal4[2]),1)# to improve since is not automatic
        enc2_All=torch.cat((encModal1[1],encModal2[1],encModal3[1],encModal4[1]),1)# to improve since is not automatic
        enc1_All=torch.cat((encModal1[0],encModal2[0],encModal3[0],encModal4[0]),1)# to improve since is not automatic
        
        
        outViT1, hidden_states_out = self.vit1(torch.cat((encModal2[3],encModal3[3]),1))
        outViT2, hidden_states_out = self.vit2(torch.cat((encModal1[3],encModal4[3]),1))

        
        
        dec4_1 = self.proj_feat(outViT1, self.hidden_size, self.feat_size)
        dec4_2 = self.proj_feat(outViT2, self.hidden_size, self.feat_size)

        
        #dec4_All=torch.cat((dec4_1,dec4_2,dec4_3,dec4_4),1)# to improve since is not automatic
        dec4_All = torch.sum(torch.stack([dec4_1,dec4_2]), dim=0)

        
        dec3 = self.decoder4(dec4_All,enc4_All)#change to only concat
        dec2 = self.decoder3(dec3, enc3_All)
        dec1 = self.decoder2(dec2, enc2_All)
        out = self.decoder1(dec1, enc1_All)
        
        return self.out(out)
    

    def name(self):
        return 'Hierarchical 4CNN + 2T'




















































    
 
