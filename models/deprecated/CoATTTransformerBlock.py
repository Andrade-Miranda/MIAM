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

import torch.nn as nn

from monai.networks.blocks.mlp import MLPBlock


class CoATTBlock(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CoATT(hidden_size, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


import torch
import torch.nn as nn

from monai.utils import optional_import

einops, _ = optional_import("einops")


class CoATT(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        k_co=[]
        v_co=[]
        kf=[]
        vf=[]
        q,k,v= einops.rearrange(self.qkv(x), "b (h m) (qkv l d) -> qkv b l m h d", qkv=3, l=self.num_heads, m=4)
            
        for j in range(k.size(2)):
            for i in range(k.size(2)):
                if i!=j:
                    k_co.append(k[:,:,i])
                    v_co.append(v[:,:,i])
            kf.append(torch.cat(k_co,-2))     
            vf.append(torch.cat(v_co,-2))
            k_co=[]
            v_co=[]
            
        att_mat1 = [(torch.einsum("blxd,blyd->blxy", q[:,:,i], kf[i]) * self.scale).softmax(dim=-1) for i in range(q.size(2))]
        att_mat = [self.drop_weights(att_mat1[i]) for i in range(len(att_mat1))]
        x1 = [torch.einsum("bhxy,bhyd->bhxd", att_mat[i], vf[i]) for i in range(q.size(2))]
        x1 = einops.rearrange(torch.cat(x1,2), "b h l d -> b l (h d)")
        x1 = self.out_proj(x1)
        x1 = self.drop_output(x1)
        return x1
