from matplotlib.pyplot import grid
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops

import numpy as np

from timm.layers.helpers import to_3tuple
from .mae3d import build_3d_sincos_position_embedding
from util.util import print_network
from util.block import LayerNorm3d,LayerNorm2d

import models.Priors
import models.Priors.networks

import time

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        # pdb.set_trace()
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x

class PatchEmbed2P1D(nn.Module):
    """ 2D + 1D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        kernel_size1 = (patch_size[0], patch_size[1], 1)
        kernel_size2 = (1, 1, patch_size[2])

        self.proj = nn.Sequential(nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size1, stride=kernel_size1),
                                  nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size2, stride=kernel_size2))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        # pdb.set_trace()
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x

class UNETR3D(nn.Module):
    """General segmenter module for 3D medical images
    """
    def __init__(self, args):
        super().__init__()
        mask_in_chans=args.mask_in_chans #setup automatically
        self.opt=args
        numberofPrior=len(args.input_prior) #setup automatically

        self.encoder = args.enc_arch(img_size=args.imageSize,
                               patch_size=args.patchSize,
                               in_chans=args.input_nc,
                               embed_dim=args.hidden_size,
                               depth=args.num_layers,
                               num_heads=args.num_heads,
                               drop_path_rate=args.drop_path_rate,
                               embed_layer=PatchEmbed3D,
                               use_learnable_pos_emb=True,
                               return_hidden_states= True
                               )
        self.decoder = args.dec_arch(in_channels=args.input_nc,
                               out_channels=args.output_nc,
                               img_size=args.imageSize,
                               patch_size=args.patchSize,
                               feature_size=args.feature_size,
                               hidden_size=args.hidden_size,
                               spatial_dims=3)
                
        self.neck = nn.Sequential(
                                nn.Conv3d(
                                        args.hidden_size,
                                        args.prompt_embed_dim,
                                        kernel_size=1,
                                        bias=False,
                                ),
                                LayerNorm3d(args.prompt_embed_dim),
                                nn.Conv3d(
                                        args.prompt_embed_dim,
                                        args.prompt_embed_dim,
                                        kernel_size=3,
                                        padding=1,
                                        bias=False,
                                ),
                                LayerNorm3d(args.prompt_embed_dim),
                                )
        
        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(numberofPrior, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans),
            nn.GELU(),
            nn.Conv3d(mask_in_chans, args.prompt_embed_dim, kernel_size=1),
        )
    
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.encoder: 'encoder',
                              self.decoder: 'decoder'}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay():
                    total_set.add(f'{prefix}.{name}')
        print(f"{total_set} will skip weight decay")
        return total_set
    
    def forward(self, x_in, priors=None, time_meters=None):
        """
        x_in in shape of [BCHWD]
        """
        s_time = time.perf_counter()
        x, hidden_states = self.encoder(x_in, time_meters=time_meters)
        #B,P,C=x.shape
        #if self.training:
        #    x = self.neck(einops.rearrange(x, "b (h w d) c -> b c h w d",h=round(P**(1/3)),w=round(P**(1/3))))
        #    x=einops.rearrange(x+self.mask_downscaling(F.interpolate(priors, scale_factor=1/4, mode="trilinear", align_corners=False)),"b c h w d -> b (h w d) c")
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['enc'].append(time.perf_counter() - s_time)

        s_time = time.perf_counter()
        logits = self.decoder(x_in, x, hidden_states)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['dec'].append(time.perf_counter() - s_time)
        return logits

    def name(self):
        return print(f"Encoder: {self.encoder.name()} + decoder: {self.decoder.name()}")
    

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
        """---------------------""" # option to load pre-trained for fine tunning
        #if not('Test' in self.opt.TrainConfig):
        #    model=init_weights(model, init_type, init_gain=init_gain)
        return model
    """--------------------------------------------------------------------""" 