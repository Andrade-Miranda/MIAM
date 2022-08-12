#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:22:28 2022

@author: gustavo
"""

# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from util.util import print_network

from .decoder import CNN_PuPMLA
from timm.models.layers import trunc_normal_




class VITNaive(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

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
        res_block=opt.res_block
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        filters_Encoder=opt.filters_Encoder
        conv_block=True
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
        super(VITNaive,self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        """ -------------------VIT encoders------------------------------- """      
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = num_layers
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        
        self.numConvLevel=len(filters_Encoder)

        
        
        self.vit = ViT(
            in_channels=in_channels,
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
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=filters_Encoder[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=filters_Encoder[1],
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=filters_Encoder[2],
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=filters_Encoder[3],
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
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
                   num_modality=1,
                   features=filters_Encoder,
                   norm_name=norm_name,
                   res_block=res_block,            
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0], out_channels=out_channels)
        """ ------------------------------------------------------------- """      


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        skip_connections=[]
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        skip_connections=[enc1,enc2,enc3,enc4]
        
        decfinal = self.proj_feat(x, self.hidden_size, self.feat_size)
        decfinal= self.UpsamplingConv(decfinal)
        j=-1
        for numdec in range(self.numConvLevel):
            decfinal = self.decoder.decoderList[numdec](decfinal,skip_connections[j])#change to only concat
            j=j-1
        
        return self.out(decfinal)


    def name(self):
        return 'VIT Naive'


        """--------------------Initialize network weights.---------------"""   
    def init_weights(self,model):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.pretrained:
            
            if isinstance(self.pretrained, str):
                self.apply(_init_weights)
                #logger = get_root_logger()
                #logger.info(f'load model from: {self.pretrained}')

                if self.pretrained2d:
                    # Inflate 2D model into 3D model.
                    self.inflate_weights()
                else:
                    # Directly load 3D model.
                    model.load_state_dict(
                    torch.load(self.pretrained,map_location=self.opt.device)
                        )
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
        
    def init_net(self,model):
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

    