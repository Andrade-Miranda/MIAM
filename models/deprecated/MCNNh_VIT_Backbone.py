#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:28:54 2021

@author: gustavo
"""


import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer,get_act_layer,get_norm_layer
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
from monai.networks.blocks.unetr_block import UnetrUpBlock
import einops
from .ViT_StreamSM import ViT_S


""" Basic Unet 
    A UNet Encoder block  implementation with 1D/2D/3D supports.
        Based on:
Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
Morphometry". Nature Methods 16, 67–70 (2019), DOI:http://dx.doi.org/10.1038/s41592-018-0261-2
    Adapted from Monai
"""
""" CNN heavy --CNN_h
"""
class EncoderBottleneck(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       in_channels,
       features,
       norm_name,
       res_block,
       conv_block,
       img_size,
       patchSize,
       mlp_dim,
       num_heads,
       num_layers,
       pos_embed,
       dropout_rate
    ):

        super(EncoderBottleneck,self).__init__()
        self.numModal=in_channels
        self.features=features
        self.patchSize=patchSize
        self.M1=nn.ModuleList()
        self.M2=nn.ModuleList()

######### modality 1 CT##########################################"       
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
            self.M1.append(encoder)

######### modality 2 PeT##########################################"              
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
            self.M2.append(encoder)       

        self.avgpool=nn.AdaptiveAvgPool3d(8)

######### ViT Fusion##########################################"              
        img_size.append(img_size[0])
        dowfactor = [ensure_tuple_rep((2**i), spatial_dims) for i in range(len(features))]
        img_sizes = [img_d // p_d for img_d, p_d in zip(img_size,dowfactor)]
        self.VitBackbone = nn.ModuleList()
        for i in range(len(features)):
            vit = ViT_S(
                numModal=self.numModal,
                in_channels=features[i],
                img_size=img_sizes[i],
                patch_size=1,
                hidden_size=features[i],
                mlp_dim=mlp_dim,
                pos_embed=pos_embed,
                num_layers=num_layers,#12
                num_heads=num_heads,#3
                classification=False,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
                )
            self.VitBackbone.append(vit)

        """ ------------------------------------------------------------- """
        scale_factor=[16,8,4,2]
        self.ProjShared = nn.ModuleList()
        for i in range(len(features)):
            ProjShared=CommonFeatureSpace(spatial_dims,
               in_channels=features[i],
               out_channels=features[i],
               norm_name=norm_name,
               scale_factor=scale_factor[i])
            self.ProjShared.append(ProjShared)

    def forward(self,x):
        skip=[]
        #xin=list(torch.split(x,1,1))
        x1=x[:,0:1].clone()
        x2=x[:,1:2].clone()      
        for j in range(len(self.features)):
            x1 = self.M1[j](x1)
            x2 = self.M2[j](x2)
            outViT,hidden_state = self.VitBackbone[j]([self.avgpool(x1),self.avgpool(x2)])
            outViT = einops.rearrange(outViT, "b (Np n) H -> b n Np H",n=self.numModal)
            decfinal=[self.proj_feat(outViT[:,i], self.features[j], ensure_tuple_rep(8,3)) for i in range(self.numModal)]
            x1,x2=self.ProjShared[j](x1,x2,decfinal)
            skip.append(torch.cat((x1,x2),1))
        return skip      

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
""""
------------------------------------------------------------------------
CommonFeatureSpace
-----------------------------------------------------------------------
"""
class CommonFeatureSpace(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       in_channels,
       out_channels,
       norm_name,
       scale_factor

    ):
        super(CommonFeatureSpace,self).__init__()
        
        self.convM1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=True,
            bias=True
        )
        
        self.convM12 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=True,
            bias=True
        )
        
        self.convM2 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=True,
            bias=True
        )
        
        self.convM21 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=True,
            bias=True
        )
        
        self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True) 
        
        self.sigmoid = get_act_layer(name='sigmoid')
        self.norm1 = get_norm_layer(name='instance', spatial_dims=spatial_dims)
        self.norm2 = get_norm_layer(name='instance', spatial_dims=spatial_dims)

    
    def forward(self,M1,M2,M12):
        fM1=self.convM1(M1)
        fhatM1=self.sigmoid(fM1)*fM1
        fM12=self.up(M12[0])
        fM12=self.convM12(fM12)
        fhatM12=self.sigmoid(fM12)*fM12
        Fm12=self.norm1(self.norm1(fhatM1)+self.norm1(fhatM12))
        
        fM2=self.convM2(M2)
        fhatM2=self.sigmoid(fM2)*fM2
        fM21=self.up(M12[1])
        fM21=self.convM21(fM21)
        fhatM21=self.sigmoid(fM21)*fM21
        Fm21=self.norm2(self.norm2(fhatM2)+self.norm2(fhatM21))
        
        return [Fm12,Fm21]

""""
------------------------------------------------------------------------
DECODER
-----------------------------------------------------------------------
"""
class CNN_PuPMLA(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
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
                    in_channels=features[-1] * num_modality,# first correspond to the hidden_size coming from transformer
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
    


  

class MultiCNNHeavy(nn.Module):

    def __init__(self, opt):
        
        in_channels=opt.input_nc
        out_channels= opt.output_nc
        img_size=opt.imageSize
        norm_name=opt.norm_name
        patchSize=opt.patchSize
        mlp_dim= opt.mlp_dim
        num_heads=opt.num_heads
        num_layers=opt.num_layers
        pos_embed=opt.pos_embed
        filters_Encoder=opt.filters_Encoder
        res_block=opt.res_block
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
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
        super(MultiCNNHeavy,self).__init__()
        
        self.numModal=in_channels #gloabal variable to keep the number of modalities

        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        
        """ -------multipath encoders------------------------------------- """
        self.encodModalities=EncoderBottleneck(
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            res_block=res_block,
            conv_block=True,
            img_size=img_size,
            patchSize=patchSize,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate
            )
        """ ----------------------------------------------------------------""" 

             
        self.MaxPool=nn.MaxPool3d(3, stride=2,padding=0,dilation=1,ceil_mode=True)

        """ -------------------CNN decoders------------------------------- """             
        self.decoder=CNN_PuPMLA(
                   spatial_dims=spatial_dims,
                   num_modality=in_channels,
                   features=filters_Encoder,
                   norm_name=norm_name,
                   res_block=res_block,            
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0] * in_channels, out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        
    
        
        self.numModal=in_channels #gloabal variable to keep the number of modalities
        self.numConvLevel=len(filters_Encoder)

        
    def forward(self,x):
        
        skip_connections=self.encodModalities(x)
        
        decfinal=self.MaxPool(skip_connections[-1])

        j=-1
        for numdec in range(self.numConvLevel):
            decfinal = self.decoder.decoderList[numdec](decfinal,skip_connections[j])#change to only concat
            j=j-1
        
        return self.out(decfinal)
    

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
        self.init_weights(model, init_type, init_gain=init_gain)
        return model
    """--------------------------------------------------------------------""" 
































