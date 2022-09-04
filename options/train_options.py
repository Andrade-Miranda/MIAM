from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--val_interval', type=int, default=1, help='# data for val')
        self.parser.add_argument('--imageSize', nargs='+', default= 0, help='Image Size after pre-processing')
        self.parser.add_argument('--epochs', type=int, default=150, help='# of epochs')
        self.parser.add_argument('--VAL_AMP',  dest='VAL_AMP', action='store_true',default=False, help='Automatic Mixed Precision package - torch.cuda.amp')
        self.parser.add_argument('--MoreAug',  dest='MoreAug', action='store_true',default=False, help='Extra Augmentation, NO AVAILABLE')
        self.parser.add_argument('--region', nargs='+', default=((1,4),(1,4,2),(4,)), help='segmentation regions to merge, default Brats')
        self.parser.add_argument('--patchSize', type=int, default=1, help='number of the patch for transformer network')
        self.parser.add_argument('--hidden_size', type=int, default=768, help='dimension of the transformer hiddensize')
        self.parser.add_argument('--mlp_dim', type=int, default=3072, help='dimension of the transformer multilayer perceptron')
        self.parser.add_argument('--num_heads', type=int, default=12, help='number of head of each transformer block')
        self.parser.add_argument('--num_layers', type=int, default=12, help='number of encoder in the transformer block')
        self.parser.add_argument('--pos_embed', type=str, default="perceptron", help='positional embedding strategy')    
        self.parser.add_argument('--norm_name', type=str, default="instance", help='normalization strategy')
        self.parser.add_argument('--pretrained', type=str, default=None, help='no use pretrained models')   
        self.parser.add_argument('--pretrained2d', dest='pretrained2d',action='store_true', default=False, help='When 2d pretraining are available')  
        self.parser.add_argument('--filters_Encoder', nargs='+', default=(16,32,64,128), help='filters for the CNN network')
        self.parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')   
        self.parser.add_argument('--res_block', dest='res_block',action='store_false', default=True, help='if is True the CNN network use resnet blocks')  
        self.parser.add_argument('--spatial_dims', type=int, default=3, help='The network will be 2D or 3D')
        self.parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',help='learning rate (default: 5e-4)')
        self.parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

####################OPTION AVAILABLE only with config contrastive###################################""""        
        self.parser.add_argument('--lambdaCNN', type=float, default=1e-1, help='lambda contrastive CNN')
        self.parser.add_argument('--lambdaViT', type=float, default=1e-2, help='lambda contrastive vit')
        self.parser.add_argument('--SupContrast',  dest='SupContrast', action='store_true',default=False, help='Supervised contrastive loss only work when config CL is used')
        self.parser.add_argument('--PatchNCELoss',  dest='PatchNCELoss', action='store_true',default=False, help='NCE patch contrastive loss only work when config CL is used')
        self.isTrain = True

        
####################OPTION AVAILABLE WITH HYBRID TIMM###################################""""
#### Training OPTIONS when use TIMM config based on DEIT paper###########################
        # Optimizer parameters
    #     self.parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
    #                         help='Optimizer (default: "adamw"')
    #     self.parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='Optimizer Epsilon (default: 1e-8)')
    #     self.parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='Optimizer Betas (default: None, use opt default)')
    #     self.parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
    #                     help='Clip gradient norm (default: None, no clipping)')
    #     self.parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    # # Learning rate schedule parameters
    #     self.parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "cosine"')
    #     self.parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    #     self.parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    #     self.parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    #     self.parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
    #                     help='warmup learning rate (default: 1e-6)')
    #     self.parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    #     self.parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
    #                     help='epoch interval to decay LR')
    #     self.parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR, if scheduler supports')
    #     self.parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    #     self.parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    #     self.parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')
###################################""""###################################""""###################################""""
