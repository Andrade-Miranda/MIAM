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
from .ViT_StreamSM import ViT_S
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network

from .decoder import CNN_PuPMLA


class VITSingle(nn.Module):
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
        super(VITSingle,self).__init__()

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

        
        self.vit = ViT_S(
            numModal=self.numModal,
            in_channels=2,
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
            in_channels=in_channels*2,
            out_channels=filters_Encoder[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size*self.numModal,
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
            in_channels=hidden_size*self.numModal,
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
            in_channels=hidden_size*self.numModal,
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
                   num_modality=1,
                   features=filters_Encoder,
                   norm_name=norm_name,
                   res_block=res_block,            
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0], out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        
    def Del_SepTOKEN(self,x):
        xout=[]
        prev=0
        w,h,c=self.feat_size
        siz=(w*h*c)
        last=siz
        for i in range(self.numModal):    
            xout.append(x[:,prev:last,:])
            prev=last+1
            last=prev+siz
        return torch.cat(xout,1)


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0),*feat_size, (hidden_size*self.numModal))
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    



    def forward(self, x_in):
        x1=torch.cat([x_in[:,0,:,:,:].unsqueeze(1),x_in[:,1,:,:,:].unsqueeze(1)],dim=1)#0,3-1,2
        x2=torch.cat([x_in[:,2,:,:,:].unsqueeze(1),x_in[:,3,:,:,:].unsqueeze(1)],dim=1)
        x=[x1,x2]
        x, hidden_states_out = self.vit(x)
        enc1 = self.encoder1(x_in)
        x2 = self.Del_SepTOKEN(hidden_states_out[3])
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = self.Del_SepTOKEN(hidden_states_out[6])
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = self.Del_SepTOKEN(hidden_states_out[9])
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        skip_connections=[enc1,enc2,enc3,enc4]
        
        decfinal = self.proj_feat(self.Del_SepTOKEN(x), self.hidden_size, self.feat_size)
        decfinal= self.UpsamplingConv(decfinal)
        j=-1
        for numdec in range(self.numConvLevel):
            decfinal = self.decoder.decoderList[numdec](decfinal,skip_connections[j])#change to only concat
            j=j-1
        
        return self.out(decfinal)


    def name(self):
        return 'VIT Single'


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
