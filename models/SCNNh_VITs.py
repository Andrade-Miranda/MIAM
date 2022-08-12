#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:26:58 2021

@author: gustavo
"""



import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock,get_conv_layer,get_act_layer,get_norm_layer
from .ViT_StreamSM import SiameseViT_S
from monai.utils import ensure_tuple_rep
from torch.nn import init
from util.util import print_network
from .decoder import CNN_PuPMLA
import einops
import numpy as np
from typing import Optional, Sequence, Tuple, Union


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
            [get_norm_layer(name=(norm_name,{"affine":True}), spatial_dims=spatial_dims, channels=out_channels) for i in range(num_modalities)]
        )
        
        self.norm2 = nn.ModuleList(
            [get_norm_layer(name=(norm_name,{"affine":True}), spatial_dims=spatial_dims, channels=out_channels) for i in range(num_modalities)]
        )
        
        self.norm3 = nn.ModuleList(
            [get_norm_layer(name=(norm_name,{"affine":True}), spatial_dims=spatial_dims, channels=out_channels) for i in range(num_modalities)]
        )        
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

    def forward(self,x,modal):
        residual = x
        out = self.conv1(x)
        out= self.norm1[modal](out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out= self.norm2[modal](out)        
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3[modal](residual)
        out += residual
        out = self.lrelu(out)
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
class SiameseEncoder(nn.ModuleList):

    def __init__(
            self,
               num_modalities,
               spatial_dims,
               in_channels,
               features,
               norm_name
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


    def forward_once(self, x,modal):
        y=[]
        for j in range(len(self.encoderList)):
            x = self.encoderList[j](x,modal)
            y.append(x)
        return y       
    
    def forward(self, input1,input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1,0)
        output2 = self.forward_once(input2,1)
        
        return output1, output2
    
    




  

class SharedCNN_VITSimple(nn.Module):

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
        
        self.numModal=in_channels #gloabal variable to keep the number of modalities
        self.numConvLevel=len(filters_Encoder)
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
        super(SharedCNN_VITSimple,self).__init__()
        
        
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        
        
        """ -------multipath encoders------------------------------------- """
        
        self.encodModalities = SiameseEncoder(
            num_modalities=in_channels,
            spatial_dims= spatial_dims,
            in_channels= in_channels,
            features= filters_Encoder,
            norm_name=norm_name,
            )
        self.MaxPool=nn.MaxPool3d(3, stride=2,padding=0,dilation=1,ceil_mode=True)
        """ ----------------------------------------------------------------"""      
       
        
        """ -------------------VIT encoders------------------------------- """      
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
            
        self.num_layers = num_layers
        downfactor=int(2**(len(filters_Encoder)))
        img_size = tuple([int((x/downfactor)) for x in img_size])
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(feature_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = SiameseViT_S(
            numModal=self.numModal,
            in_channels=filters_Encoder[-1],
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            pos_embed=pos_embed,
            num_layers=self.num_layers,
            num_heads=num_heads,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        """ ------------------------------------------------------------- """  
        self.ProjShared=CommonFeatureSpace(spatial_dims,
               in_channels=hidden_size,
               out_channels=hidden_size)
        
        
        """ -------------------CNN decoders------------------------------- """
        self.UpsamplingConv=get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            stride=2,
            conv_only=True,
            is_transposed=True,
            )
             
        self.decoder=CNN_PuPMLA(
                   spatial_dims=spatial_dims,
                   hidden_size=hidden_size,
                   num_modality=in_channels,
                   features=filters_Encoder,
                   norm_name=norm_name,
                   res_block=res_block,            
                   )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=filters_Encoder[0] * in_channels, out_channels=out_channels)
        """ ------------------------------------------------------------- """      
        
    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    

    def forward(self, x_in):
        
        z1,z2=self.encodModalities(x_in[:,0:1],x_in[:,1:2])#modify for multiples modalities
        
        skip_connections=[]
        for i in range(self.numConvLevel):
            skip_connections.append(torch.cat((z1[i],z2[i]),1))
        
        lastConv=[]# last conv i downsampling using maxpooling
        for modal in [z1[-1],z2[-1]]:
            lastConv.append(self.MaxPool(modal))# list of tensor
        
        # change for more than two modalities
        outViT, hidden_states_out = self.vit(lastConv)# space latent that can be used for contrastive learning
        outViT2Mod = einops.rearrange(outViT[-1], "b (Np n) H -> b n Np H",n=self.numModal)
        decfinal=[self.proj_feat(outViT2Mod[:,i], self.hidden_size, self.feat_size) for i in range(self.numModal)]
        decfinal=self.ProjShared(self.proj_feat(outViT[0], self.hidden_size, self.feat_size),self.proj_feat(outViT[1],self.hidden_size, self.feat_size),decfinal)
        decfinal= self.UpsamplingConv(decfinal)
        
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
        """---------------------"""
        self.init_weights(model, init_type, init_gain=init_gain)
        return model
    """--------------------------------------------------------------------""" 





class CommonFeatureSpace(nn.ModuleList):

    def __init__(
       self,
       spatial_dims,
       in_channels,
       out_channels,
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
        
        self.convM2 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            conv_only=True,
            bias=True
        )
        
        self.sigmoid = get_act_layer(name='sigmoid')
        self.norm3 = get_norm_layer(name=('layer',{"normalized_shape":out_channels}), spatial_dims=spatial_dims)
    
    def forward(self,M1,M2,M12):
        fM1=self.convM1(M12[0])
        fM2=self.convM2(M12[1])
        fhatM1=self.sigmoid(fM1)*M1
        fhatM2=self.sigmoid(fM2)*M2
        fhatM1 = fhatM1.permute(0, 2, 3,4, 1)
        fhatM2 = fhatM2.permute(0, 2, 3,4, 1)
        Fm12=self.norm3(self.norm3(fhatM1)+self.norm3(fhatM2))
        Fm12 = Fm12.permute(0, 4, 1, 2,3)
        
        return Fm12









































