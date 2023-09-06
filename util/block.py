#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional
from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.dynunet_block import get_conv_layer,get_act_layer,get_norm_layer
import numpy as np


######FUSION AFTER VIT interaction#################################"
class FusedGatedUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(FusedGatedUnit, self).__init__()
        #self.fc_embeddings = nn.ModuleList()
        #for i in range(num_modalities):
        #    self.fc_embeddings.append(nn.Linear(input_dimension, output_dimension))
        self.fc_embeddings =nn.Linear(input_dimension, output_dimension)# only one embedding for all modalities
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        xin=[]
        for i in range(x.shape[1]):
            xin.append(self.fc_embeddings(x[:,i]))
        x = torch.stack(xin, dim=0).sum(dim=0)
        x = self.cg(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return functional.glu(x, 1)


class CommonFeatureSpace(nn.ModuleList):
    def __init__(
       self,
       spatial_dims,
       in_channels,
       out_channels,
       norm_name

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
        self.norm1 = get_norm_layer(name='instance', spatial_dims=spatial_dims)
        self.norm2 = get_norm_layer(name='instance', spatial_dims=spatial_dims)
        self.norm3 = get_norm_layer(name='instance', spatial_dims=spatial_dims)
    
    def forward(self,M1,M2,M12):
        fM1=self.norm1(self.convM1(torch.cat((self.norm1(M1),self.norm1(M12[0])),1)))
        fM2=self.norm2(self.convM2(torch.cat((self.norm2(M1),self.norm2(M12[1])),1)))
        fhatM1=self.sigmoid(fM1)*fM1
        fhatM2=self.sigmoid(fM2)*fM2

        Fm12=self.norm3(self.norm3(fhatM1)+self.norm3(fhatM2))
        
        return Fm12
#################################"#################################"#################################"

    



######################RESNET####################################
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


class UPResBlock(nn.Module):
    """
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
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,#to check
            stride=stride,
            conv_only=True,
            is_transposed=True,
        )
        self.resblock=UnetResBlock(num_modalities,
               spatial_dims=spatial_dims,
               in_channels=out_channels*2,
               out_channels=out_channels,
               kernel_size=kernel_size,
               stride=1,
               norm_name=norm_name,
               dropout=dropout)

    def forward(self, inp,skip,modal):
        out = self.transp_conv(inp)
        out=torch.cat([out,skip],dim=1)
        out=self.resblock(out,modal)
        

        return out

##########################################################

########################################RES+SENORM############################
class UnetResBlockNormSE(nn.Module):
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
            [FastSmoothSENorm(in_channels=out_channels) for i in range(num_modalities)]
        )
        
        self.norm2 = nn.ModuleList(
            [FastSmoothSENorm(in_channels=out_channels) for i in range(num_modalities)]
        )
        
        self.norm3 = nn.ModuleList(
            [FastSmoothSENorm(in_channels=out_channels) for i in range(num_modalities)]
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


class UPResBlockNormSE(nn.Module):
    """
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
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,#to check
            stride=stride,
            conv_only=True,
            is_transposed=True,
        )
        self.resblock=UnetResBlockNormSE(num_modalities,
               spatial_dims=spatial_dims,
               in_channels=out_channels*2,
               out_channels=out_channels,
               kernel_size=kernel_size,
               stride=1,
               norm_name=norm_name,
               dropout=dropout)

    def forward(self, inp,skip,modal):
        out = self.transp_conv(inp)
        out=torch.cat([out,skip],dim=1)
        out=self.resblock(out,modal)
        
        return out
################################################################################




################BLOCKS SQUEEZE AND EXCITATION##################################
class FastSmoothSENorm(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=2):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = functional.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super(FastSmoothSENorm, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class FastSmoothSeNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(FastSmoothSeNormConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = FastSmoothSENorm(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = functional.relu(x, inplace=True)
        x = self.norm(x)
        return x


class RESseNormConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super().__init__()
        self.conv1 = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, **kwargs)

        if in_channels != out_channels:
            self.res_conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = FastSmoothSeNormConv3d(in_channels, out_channels, reduction, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = functional.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x






###################################PH######################################
class SingleConv(nn.Module):
    ''' {Conv2d, BN, ReLU} '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.single_conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.single_conv(x)

class SimpleConv(nn.Module):
    ''' {Conv2d, BN, ReLU} '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.simple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.simple_conv(x)
    
class DoubleConv(nn.Module):
    ''' {Conv2d, BN, ReLU}x2 '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.double_conv(x)
    
class TripleConv(nn.Module):
    ''' {Conv2d, BN, ReLU}x3 '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))          

    def forward(self, x):
        return self.triple_conv(x)
    
class QuadripleConv(nn.Module):
    ''' {Conv2d, BN, ReLU}x4 '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.quadriple_conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.quadriple_conv(x)
    
class DoubleDown(nn.Module):
    ''' maxPool2d + {Conv2d, BN, ReLU}x2 '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.double_down = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(in_chan, out_chan))

    def forward(self, x):
        return self.double_down(x)
    
class DoubleUp(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x2 '''
    
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = DoubleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
class TripleUp(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x3 '''
    
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = TripleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
class QuadripleUp(nn.Module):
    ''' ConvTranspose2d + {Conv2d, BN, ReLU}x4 '''
    
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_chan , out_chan, kernel_size=2, stride=2)
        if mid_chan == None:
            mid_chan = in_chan
        self.conv = QuadripleConv(mid_chan, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    ''' Conv2d '''
    
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
def up_sample2d(x, t, mode="bilinear"):
    ''' 2D up-sampling '''
    
    return functional.interpolate(x, t.size()[2:], mode=mode, align_corners=False)

class MixBlock(nn.Module):
    ''' for attention purposes '''
    
    def __init__(self, in_chan, out_chan):
        super(MixBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan // 4, 3, padding=1)
        self.conv3 = nn.Conv2d(in_chan, out_chan // 4, 5, padding=2)
        self.conv5 = nn.Conv2d(in_chan, out_chan // 4, 7, padding=3)
        self.conv7 = nn.Conv2d(in_chan, out_chan // 4, 9, padding=4)
        self.bn1 = nn.BatchNorm2d(out_chan // 4)
        self.bn3 = nn.BatchNorm2d(out_chan // 4)
        self.bn5 = nn.BatchNorm2d(out_chan // 4)
        self.bn7 = nn.BatchNorm2d(out_chan // 4)
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, x):
        k1 = self.bn1(self.conv1(x))
        k3 = self.bn3(self.conv3(x))
        k5 = self.bn5(self.conv5(x))
        k7 = self.bn7(self.conv7(x))
        return self.nonlinear(torch.cat((k1, k3, k5, k7), dim=1))
    
class Attention(nn.Module):
    ''' attention modules '''
    
    def __init__(self, in_chan, out_chan):
        super(Attention, self).__init__()
        self.mix1 = MixBlock(in_chan, out_chan)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.mix2 = MixBlock(out_chan, out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_chan)
        self.norm2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        mix1 = self.conv1(self.mix1(x))
        mix2 = self.mix2(mix1)
        att_map = torch.sigmoid(self.conv2(mix2))
        out = self.norm1(x*att_map) + self.norm2(shortcut)
        return self.relu(out), att_map
