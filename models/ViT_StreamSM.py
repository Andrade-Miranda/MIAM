#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:37:10 2021

@author: gustavo
"""

from typing import Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks.transformerblock import TransformerBlock
from .PatchModalEmb import PatchModalEmbBlock
from .CoATTTransformerBlock import CoATTBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from .Muti_Transformer import MultiTransformerBlock



class SiameseViT_S(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        numModal: int,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.numModal=numModal
        self.classification = classification
        self.patch_embedding = nn.ModuleList(
                        [PatchModalEmbBlock(
                                numModal=numModal,
                                in_channels=in_channels,
                                img_size=img_size,
                                patch_size=patch_size,
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                pos_embed=pos_embed,
                                dropout_rate=dropout_rate,
                                spatial_dims=spatial_dims,
                                )for i in range(numModal)]
                        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
    
    def forwardOnebyOne(self,inputsToken):
        hidden_states_out = []
        outputToken=[]
        for token in inputsToken:
            for blk in self.blocks:
                token = blk(token)
                hidden_states_out.append(token)
            outputToken.append(self.norm(token))
            
        return outputToken, hidden_states_out# solo devuelvo el hiddenstates luego de la ultima iteracion


    def forward(self, x):
        inputsToken=[]
        for i in range(self.numModal):
            inputsToken.append(self.patch_embedding[i](x[i]))# check for the case of more than two modalities
        inputsToken.append(torch.cat(inputsToken,1))
        outputs,hidden_states_out=self.forwardOnebyOne(inputsToken)
        return outputs,hidden_states_out
        









class ViT_S(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        numModal: int,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        fusion: str="Concatenation"
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        
        self.fusion=fusion
        self.numModal=numModal
        self.classification = classification
        self.patch_embedding = nn.ModuleList(
                        [PatchModalEmbBlock(
                                numModal=numModal,
                                in_channels=in_channels,
                                img_size=img_size,
                                patch_size=patch_size,
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                pos_embed=pos_embed,
                                dropout_rate=dropout_rate,
                                spatial_dims=spatial_dims,
                                modality=i
                                )for i in range(numModal)]
                        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())

    def forward(self, x):
        inputTok=[]
        for i in range(self.numModal):
            inputTok.append(self.patch_embedding[i](x[i]))
            
        if self.fusion=="Concatenation":
            x=torch.cat(inputTok,1)
        elif self.fusion=="Suma":
            x=torch.stack(inputTok, dim=0).sum(dim=0)
        else:
            raise ValueError("Option no available")
            
        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class ViT_M(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """
    def __init__(
        self,
        numModal: int,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = nn.ModuleList(
                        [PatchEmbeddingBlock(
                            in_channels=in_channels,
                            img_size=img_size,
                            patch_size=patch_size,
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            pos_embed=pos_embed,
                            dropout_rate=dropout_rate,
                            spatial_dims=spatial_dims,
                        )for i in range(numModal)]
                        )
        self.blocks = nn.ModuleList(
            [CoATTBlock(hidden_size, mlp_dim, num_heads, dropout_rate) if (i+1)%2 ==0 else MultiTransformerBlock(numModal,hidden_size, mlp_dim, num_heads, dropout_rate) for i in range(num_layers+1)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            
        self.numModal=numModal

    def forward(self, x):
        mod=0
        for layer,i in enumerate(x):
            x[mod] = self.patch_embedding[layer](i)
            mod=mod+1
        x=torch.cat(x,1)
     
        if self.classification:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if self.classification:
            x = self.classification_head(x[:, 0])
        return x, hidden_states_out