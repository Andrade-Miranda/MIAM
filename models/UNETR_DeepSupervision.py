

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
import util.block as block
import torch
from util.util import print_network
from torch.nn import init


class UNETR_DeepSupervision(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(self,opt):

        in_channels=opt.input_nc
        out_channels= opt.output_nc
        img_size=opt.imageSize
        feature_size=opt.feature_size
        hidden_size=opt.hidden_size
        mlp_dim= opt.mlp_dim
        num_heads=opt.num_heads
        pos_embed=opt.pos_embed
        norm_name=opt.norm_name
        res_block=opt.res_block
        dropout_rate= opt.dropout_rate
        spatial_dims= opt.spatial_dims
        conv_block= True
        self.opt=opt
        attention= opt.attention
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

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        
        self.attention=attention

        # COMMON ENCODER
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
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
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
            out_channels=feature_size * 4,
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
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        
        
        # FIRST DECODER
        self.decoder4_1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3_1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2_1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1_1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        
        self.down4_1=block.SimpleConv(8 * feature_size, feature_size)# output of encoder5 and stage3
        self.down3_1=block.SimpleConv(4 * feature_size, feature_size)
        self.down2_1=block.SimpleConv(2 * feature_size, feature_size)
        self.down1_1=block.SimpleConv(feature_size, feature_size)
        
        self.down_out4_1=block.SingleConv(feature_size,out_channels)
        self.down_out3_1=block.SingleConv(feature_size,out_channels)
        self.down_out2_1=block.SingleConv(feature_size,out_channels)
        self.down_out1_1=block.SingleConv(feature_size,out_channels)
        
        if self.attention:
            self.mix4_1=block.Attention(feature_size, feature_size)
            self.mix3_1=block.Attention(feature_size, feature_size)
            self.mix2_1=block.Attention(feature_size, feature_size)
            self.mix1_1=block.Attention(feature_size, feature_size)
            
            self.mix_out4_1=block.SimpleConv(feature_size, out_channels)
            self.mix_out3_1=block.SimpleConv(feature_size, out_channels)
            self.mix_out2_1=block.SimpleConv(feature_size, out_channels)
            self.mix_out1_1=block.SimpleConv(feature_size, out_channels)
        
        self.out1_1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size*4, out_channels=feature_size)
        self.out2_1 = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x


    def forward(self, x_in,DeppSuper):
        # COMMON ENCODER
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))

        # BOTTLENECK
        dec4 = self.proj_feat(x)
        
        # DECODER 1
        dec4_1 = self.decoder4_1(dec4, enc4)
        dec3_1 = self.decoder3_1(dec4_1, enc3)
        dec2_1 = self.decoder2_1(dec3_1, enc2)
        dec1_1 = self.decoder1_1(dec2_1, enc1)
        
        #supervision
        down4_1=block.up_sample3d(self.down4_1(dec4_1), x_in)
        down3_1=block.up_sample3d(self.down3_1(dec3_1), x_in)
        down2_1=block.up_sample3d(self.down2_1(dec2_1), x_in)
        down1_1=self.down1_1(dec1_1)
        
        down_out4_1=self.down_out4_1(down4_1)
        down_out3_1=self.down_out3_1(down3_1)
        down_out2_1=self.down_out2_1(down2_1)
        down_out1_1=self.down_out1_1(down1_1)
        
        if self.attention:
            mix4_1, att4_1= self.mix4_1(down4_1)
            mix3_1, att3_1= self.mix3_1(down3_1)
            mix2_1, att2_1= self.mix2_1(down2_1)
            mix1_1, att1_1= self.mix1_1(down1_1)
            
            mix_out4_1=self.mix_out4_1(mix4_1)
            mix_out3_1=self.mix_out3_1(mix3_1)
            mix_out2_1=self.mix_out2_1(mix2_1)
            mix_out1_1=self.mix_out1_1(mix1_1)
            logits1=self.out2_1(self.out1_1(torch.cat((mix4_1,mix3_1,mix2_1,mix1_1),dim=1)))
        else:
            logits1=self.out2_1(self.out1_1(torch.cat((down4_1,down3_1,down2_1,down1_1),dim=1)))

               
        if DeppSuper:
            if self.attention:
                return [down_out4_1,mix_out4_1,down_out3_1,mix_out3_1,down_out2_1,mix_out2_1,down_out1_1,mix_out1_1,logits1]
            else:
                return [down_out4_1,down_out3_1,down_out2_1,down_out1_1,logits1]
        else:
            return logits1
        
    def name(self):
        return 'Unet Transformer'
    
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
        if self.opt.pretrained:
            if isinstance(self.opt.pretrained, str):
                model.load_state_dict(torch.load(self.opt.pretrained,map_location=self.opt.device),strict=False)
                print('initialize network with pretained weights %s' % self.opt.pretrained)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            model=self.init_weights(model, init_type, init_gain=init_gain)
        
        return model
    """--------------------------------------------------------------------""" 

if __name__ == '__main__':
    from torchsummary import summary
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNETR_DeepSupervision(in_channels=1, out_channels=1, img_size=(96,96,96), feature_size=16, hidden_size=768, mlp_dim=3072, 
                             num_heads=12, pos_embed='perceptron', norm_name='instance', res_block=True, 
                             dropout_rate=0, spatial_dims=3, attention=True).to(device)

    inputs = torch.rand(1,1,96,96,96)
    pred1, pred2 = model(inputs.to(device))

    print('WORKING!!!!')

    
