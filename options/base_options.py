import argparse
import os
from util import util
import torch
from util.visualizer import VisualPlots
from util.nnUNetUtils import nnUNETPlanning
from torch.utils.tensorboard import SummaryWriter
from util.loggings import WandbLogger,get_rank
import numpy as np
import wandb
import sys
from omegaconf import OmegaConf

#os.environ["WANDB_MODE"]="offline"



class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        conf_file = sys.argv[1]
        if os.path.exists(conf_file):
            print(f"Config file {conf_file} exist!")
             ###config file ######
            self.configFile = OmegaConf.load(conf_file)
            self.parser.add_argument('conf_file', type=str, help='path to config file')       
        #####################

        ####Base setting###
        self.parser.add_argument('--dataroot', type=str,default='Task001_BraTS2021', help='dataset path (Task001_Prostate, json file "./datasets/BraTS2021/dataset.json") or Folder with images, it will depend of the configuration')
        self.parser.add_argument('--Val_batchSize', type=int, default=1, help='validation batch size')
        self.parser.add_argument('--val_interval', type=int, default=1, help='# interval to do the evaluation')
        self.parser.add_argument('--validate_min_epoch', type=int, default=1, help='# value to start the evaluation')
        self.parser.add_argument('--max_num_threads', type=int, default=12, help='# value max of thread')
        self.parser.add_argument('--update_freq', type=int, default=1, help='# gradient accumulation steps')
        self.parser.add_argument('--batchSize', type=int, default=0, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--TrainConfig', type=str, default='BaseConfig',help='Configuration file that specify optimizers, metrics, lr schedule, etc')
        self.parser.add_argument('--Deterministic', dest='Deterministic',action='store_true',default=False, help='if is True the Deterministic training for reproducibility')          
        self.parser.add_argument('--loadsplit',  type=str, default=None,help='load custom splits saved in splits_plk file')  
        self.parser.add_argument('--checkpoints_dir', type=str, default=None, help='models are saved here, default is None meaning that files will save in ./checkpoints/TaskName')
        self.parser.add_argument('--debug', dest='debug',action='store_true', default=False, help='save the batch input image')#set as debug
        self.parser.add_argument('--yh_run_model', type=str, default='Train',choices=('Train','Continue'), help='chooses which Train or continue')#no used yet by the moment test and training has different scripts
        self.parser.add_argument('--dataset_mode', type=str, default='nnUNet', help='choose the dataset mode to load the data, by default BRATS')
        self.parser.add_argument('--output_dir', type=str, default=None, help='save test segmentatio output results here, default is None meaning that files will save in ./Output/TaskName')
        self.parser.add_argument('--.', type=int, default=0, help='custom_sub_dir')

        # models
        self.parser.add_argument('--encoder', type=str, default='MCNN_h+VIT_n',help='chooses encoder to use CNN_h+VIT_n, CNN_l+VIT_n,MCNN_{h,l}+VIT_{n,s,m}')        
        self.initialized = True
        
        #nnUnet setting
        self.parser.add_argument('--plan', type=str, default="nnUNetPlansv2.1_plans_3D.pkl",help='plan from NNunet-Preprocessing')        
        self.parser.add_argument('--fold', type=int, default=0, help='choose number of fold used to train data')
        self.parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for the cross-validation')

        # Test setting
        self.parser.add_argument('--sigmoid', dest='sigmoid',action='store_true', default=False, help='helping bool to change the default post processing setup for testing')  
        self.parser.add_argument('--postprocessing', type=str, default="None",help='option for postprocessing data') 
            
    def str2None(self,v):
        """
        Converts string to None type; enables command line 
        arguments in the format of '--arg1 true --arg2 false'
        """
        if v.lower() in ('none', 'NONE', 'NoNE','None'):
            return None
        else:
            return v

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        #update opt with config file if it exist############
        if self.opt.conf_file:
            for key in self.configFile:
                setattr(self.opt, key, self.configFile[key])
        ####################################################
        
        features=(32,64,128,256,320,512,768,1028,1028,1028)#features filter for CNN network encoder

        #### Train o test ############
        if self.opt.yh_run_model=='Train' or self.opt.yh_run_model=='Continue':
            self.opt.isTrain = True   # test is not available yet
        else:
            self.opt.isTrain = False
        #############################
        
        #### device CPU or CUDA############
        if self.opt.gpu_ids =='-1':
            self.opt.device=torch.device("cpu") 
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else: 
            self.opt.device=torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        ########################

        ##############CPU specifications###""""""""
        # os.sched_getaffinity(0) is not supported by all operating systems
        try:
            self.opt.num_threads = np.min([len(os.sched_getaffinity(0)), self.opt.max_num_threads])
        except:
            self.opt.num_threads= self.opt.max_num_threads
        print(f" Total number of Threads: {self.opt.num_threads}",flush=True)
        ###################"#######################
        
        str_ids = str(self.opt.gpu_ids).split(',') #ensure that is string
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        if self.opt.pretrained!=None:
            self.opt.pretrained=os.path.join("./pretrained_ckpt",self.opt.pretrained)
        #self.args = vars(self.opt)# move to this place
        
        '------------ Load nnUNet planning -------------'
        planner=nnUNETPlanning(self.opt)
        CurrentPlan=planner.load_my_plans()
        #take always FULLRES
        self.opt.stage=len(CurrentPlan['plans_per_stage'])-1
        if len(self.opt.plan.split('_'))>3:
            self.opt.planning_stage='nnUNetData_plans_v2.1_'+self.opt.plan.split('_')[1]+'_stage'+str(self.opt.stage)
        else:
            self.opt.planning_stage='nnUNetData_plans_v2.1_stage'+str(self.opt.stage)

        self.opt.num_pool_per_axis=CurrentPlan['plans_per_stage'][self.opt.stage]['num_pool_per_axis']
        self.opt.pool_op_kernel_sizes=[[1,1,1]]+CurrentPlan['plans_per_stage'][self.opt.stage]['pool_op_kernel_sizes'] 
        self.opt.conv_kernel_sizes=CurrentPlan['plans_per_stage'][self.opt.stage]['conv_kernel_sizes'] 
        
        if self.opt.imageSize==0: # use by default nnUNet configuration : pooling, kernel and others
            self.opt.imageSize=CurrentPlan['plans_per_stage'][self.opt.stage]['patch_size'].tolist()
            self.opt.filters_Encoder=features[:len(self.opt.conv_kernel_sizes)]
        else:# configuracion for TransBTS 
            self.opt.num_pool_per_axis=[4,4,4,4]#[2,5,5]
            self.opt.pool_op_kernel_sizes=[[1,1,1],[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],[2, 2, 2]] #[[1,1,1],[1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
            self.opt.conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]]#[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3],[3, 3, 3],[3, 3, 3]]
            self.opt.filters_Encoder=(16,32,64,128,256,320)#features[:len(self.opt.conv_kernel_sizes)]

        if self.opt.batchSize==0:# use by default nnUNet batchsize configuration batchsize 
            self.opt.batchSize=CurrentPlan['plans_per_stage'][self.opt.stage]['batch_size']
            self.opt.Val_batchSize=CurrentPlan['plans_per_stage'][self.opt.stage]['batch_size']   
        '-------------'
                
        self.opt.sched=self.str2None(self.opt.sched)

        # save to the disk
        if self.opt.name is None:
            self.opt.name=self.opt.encoder+'F'+str(self.opt.fold)
        else:
            self.opt.name=self.opt.encoder+'__'+self.opt.name+'F'+str(self.opt.fold)
            
        #self.opt.imageSize=[int(self.opt.imageSize[i]) for i in range(len(self.opt.imageSize))]
        self.opt.filters_Encoder=tuple([int(self.opt.filters_Encoder[i]) for i in range(len(self.opt.filters_Encoder))])
        if self.opt.region[0]!='None' and ('BraTS' not in self.opt.dataroot):
            self.opt.region=tuple([tuple([int(i) for i in x.split(',')]) if len(x)>1 else (int(x),) for x in self.opt.region])
       
        ### set checkpoint and output folder
        if self.opt.checkpoints_dir is not None:
            expr_dir = os.path.join('./checkpoints',self.opt.dataroot,self.opt.checkpoints_dir,self.opt.encoder,self.opt.name)
        else:
            expr_dir = os.path.join('./checkpoints',self.opt.dataroot,self.opt.encoder,self.opt.name)
        util.mkdirs(expr_dir)
        
        if self.opt.output_dir is not None:
            out_dir = os.path.join('./Output',self.opt.dataroot,self.opt.output_dir,self.opt.encoder,self.opt.name)
        else:
            out_dir = os.path.join('./Output',self.opt.dataroot,self.opt.encoder,self.opt.name)
        util.mkdirs(out_dir)
        ###
        self.opt.expr_dir=expr_dir
        self.opt.out_dir=out_dir

        ########## convert namespace to dictionary#########################
        self.args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(self.args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        ###########################################################################

        ###########save config as a text file ############################### Deprecated in future version
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(self.args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        ###########save config as a xml file ############################### Deprecated in future version
        file_name = os.path.join(expr_dir, 'opt.yaml')
        # manual fixed errors caused by args when saving as yaml 
        self.args['device']=str(self.opt.device) 
        self.args['num_threads']=int(self.opt.num_threads)
        ##############################################
        confOutput = OmegaConf.create(self.args)
        with open(file_name, 'wt') as fp:
            OmegaConf.save(config=confOutput, f=fp)
        #####################################################################


        root_dir=self.opt.expr_dir
        max_epochs = self.opt.epochs
        val_interval = self.opt.val_interval
        Plots=VisualPlots(self.opt) #I will use to save some segmentation results. At the moment is only for plot loss curve
        
        # check this part
        global_rank = get_rank()

        # deprecated only use wandb
        #if global_rank == 0 and self.opt.out_dir is not None:
        #    log_dir=os.makedirs(os.path.join(self.opt.out_dir, 'logging'), exist_ok=True)
        #    self.opt.log_writer = SummaryWriter(log_dir=log_dir)
        #else:
        #    self.opt.log_writer = None

        if global_rank == 0 and self.opt.enable_wandb:
            dir_wandb=os.makedirs(os.path.join('wandb'), exist_ok=True)
            self.opt.wandb_logger = wandb.init(project=self.opt.project,
                                               entity="xamus86",
                                               config=self.opt,
                                               name=self.opt.nameRun,
                                               dir=dir_wandb)
        else:
            self.opt.wandb_logger = None
        
        
        
        
        
        return self.opt,root_dir,max_epochs,val_interval,Plots
