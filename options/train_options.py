from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--imageSize', type=int, nargs=3, default= 0, help='Image Size after pre-processing')
        self.parser.add_argument('--epochs', type=int, default=300, help='# of epochs')
        self.parser.add_argument('--num_training_steps_per_epoch', type=int, default=250, help='# of steps per epoch')
        self.parser.add_argument('--num_validation_steps_per_epoch', type=int, default=100, help='# of steps per epoch')
        self.parser.add_argument('--VAL_AMP',  dest='VAL_AMP', action='store_true',default=False, help='Automatic Mixed Precision package - torch.cuda.amp')
        self.parser.add_argument('--seed', type=int, default=12345, help='# of seed for deterministic training')
        self.parser.add_argument('--region', nargs='+', default=((1,4),(1,4,2),(4,)), help='segmentation regions to merge, default Brats')
        self.parser.add_argument('--spatial_dims', type=int, default=3, help='The network will be 2D or 3D')
        self.parser.add_argument('--oversample_foreground_percent', type=float, default=0.66, help='sampling strategy .66 fg rest uniform, for nnUNet set to 0.33')

     # transformers setting   
        self.parser.add_argument('--patchSize', type=int, default=1, help='number of the patch for transformer network')
        self.parser.add_argument('--hidden_size', type=int, default=768, help='dimension of the transformer hiddensize')
        self.parser.add_argument('--mlp_dim', type=int, default=3072, help='dimension of the transformer multilayer perceptron')
        self.parser.add_argument('--num_heads', type=int, default=12, help='number of head of each transformer block')
        self.parser.add_argument('--num_layers', type=int, default=12, help='number of encoder in the transformer block')
        self.parser.add_argument('--pos_embed', type=str, default="conv", help='positional embedding strategy')    
        self.parser.add_argument('--norm_name', type=str, default="instance", help='normalization strategy')
        self.parser.add_argument('--pretrained', type=str, default=None, help='no use pretrained models')   
        self.parser.add_argument('--pretrained2d', dest='pretrained2d',action='store_true', default=False, help='When 2d pretraining are available')  
        self.parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate')   
        self.parser.add_argument('--Earlyfusion', type=str, default="Concatenation", help='type of early fusion')
    
    # CNN setting
        self.parser.add_argument('--res_block', dest='res_block',action='store_false', default=True, help='if is True the CNN network use resnet blocks')  
        self.parser.add_argument('--filters_Encoder', nargs='+', default=(16,32,64,128), help='filters for the CNN network')

    
    # Optimization parameters
        self.parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
        self.parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
        self.parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
        self.parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
        self.parser.add_argument('--momentum', type=float, default=0.99, metavar='M',
                        help='SGD momentum (default: 0.99)')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
        self.parser.add_argument('--amsgrad', dest='amsgrad',action='store_true', default=False,help='amsgrad (default: False)')
        self.parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Learning rate schedule parameters
        self.parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
        self.parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
        self.parser.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
        self.parser.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
        self.parser.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
        self.parser.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
        self.parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
        self.parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
        self.parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
        self.parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
        self.parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
        self.parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
        self.parser.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
        self.parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
        self.parser.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
        self.parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
        self.parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
        self.parser.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
        self.parser.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
        self.parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', ## warmup-epochs is for TIMM
                   help='epochs to warmup LR, if scheduler supports')
        self.parser.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
        self.parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        self.parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
        self.parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')
    
        
    # distributed training parameters
        self.parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
        self.parser.add_argument('--local_rank', default=-1, type=int)
        self.parser.add_argument('--dist_on_itp', action='store_true',dest='dist_on_itp', default=False)
        self.parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        
    # EMA related parameters
        self.parser.add_argument('--model_ema', action='store_true', default=False)
        self.parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
        self.parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
        self.parser.add_argument('--model_ema_eval', action='store_true', default=False, help='Using ema to eval during training.')
        
        
    # Weights and Biases arguments
        self.parser.add_argument('--enable_wandb',action='store_true', dest='enable_wandb', default=False,
                    help="enable logging to Weights and Biases")
        self.parser.add_argument('--project', default='csPcATransformer', type=str,
                    help="The name of the W&B project where you're sending the new run.")
        self.parser.add_argument('--wandb_ckpt', action='store_true',dest='wandb_ckpt', default=False,
                       help="Save model checkpoints as W&B Artifacts.")
        self.parser.add_argument('--nameRun', default='UNETR', type=str,
                    help="The name of the new run.")

####################OPTION FOR LOSS function###################################""""
        self.parser.add_argument('--loss_option', type=str, default="DiceFocalLoss", help='choose loss function')        
        self.parser.add_argument('--lambda_Loss', type=str, default=['1 1'], help='lambda for compose loss, vector represented the weight of the loss')

####################OPTION AVAILABLE only with config contrastive###################################""""        
        self.parser.add_argument('--lambdaCNN', type=float, default=1e-1, help='lambda contrastive CNN')
        self.parser.add_argument('--lambdaViT', type=float, default=1e-2, help='lambda contrastive vit')
        self.parser.add_argument('--SupContrast',  dest='SupContrast', action='store_true',default=False, help='Supervised contrastive loss only work when config CL is used')
        self.parser.add_argument('--PatchNCELoss',  dest='PatchNCELoss', action='store_true',default=False, help='NCE patch contrastive loss only work when config CL is used')

        
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
