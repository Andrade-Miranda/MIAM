import argparse
import os
from util import util
import torch
from util.visualizer import VisualPlots
from util.nnUNetUtils import nnUNETPlanning


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str,default='Task001_BraTS2021', help='dataset path (Task001_Prostate, json file "./datasets/BraTS2021/dataset.json") or Folder with images, it will depend of the configuration')
        self.parser.add_argument('--Val_batchSize', type=int, default=1, help='validation batch size')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--TrainConfig', type=str, default='BaseConfig',help='Configuration file that specify optimizers, metrics, lr schedule, etc')
        self.parser.add_argument('--Deterministic', dest='Deterministic',action='store_false',default=True, help='if is True the Deterministic training for reproducibility')          
        self.parser.add_argument('--loadsplit',  type=str, default=None,help='load custom splits saved in splits_plk file')  
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--display_id', type=int, default=1, help='Display final pdf results')#no used yet
        self.parser.add_argument('--yh_run_model', type=str, default='Train', help='chooses which Train or Test')
        self.parser.add_argument('--dataset_mode', type=str, default='nnUNet', help='choose the dataset mode to load the data, by default BRATS')
        self.parser.add_argument('--output_dir', type=str, default='./Output', help='save test segmentatio output results')
        self.parser.add_argument('--.', type=int, default=0, help='custom_sub_dir')

        # models
        self.parser.add_argument('--hybrid', dest='hybrid',action='store_true', default=False, help='Encoder is hybrid, default true')  
        self.parser.add_argument('--encoder', type=str, default='MCNN_h+VIT_n',help='chooses encoder to use CNN_h+VIT_n, CNN_l+VIT_n,MCNN_{h,l}+VIT_{n,s,m}')        
        self.parser.add_argument('--decoder', type=str, default='CNN_PUP+MLA',help='chooses decoder; linear, CNN-PUP+MLA, VIT_PUP+MLA, VIT')  

        self.initialized = True
        
        #nnUnet setting
        self.parser.add_argument('--plan', type=str, default="nnUNetPlansv2.1_plans_3D.pkl",help='plan from NNunet-Preprocessing')        
        self.parser.add_argument('--fold', type=int, default=0, help='choose number of fold used to train data')
        self.parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for the cross-validation')

        

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        
        #### device CPU or CUDA############
        if self.opt.gpu_ids =='-1':
            self.opt.device=torch.device("cpu") 
        else: 
            self.opt.device=torch.device("cuda")
        ########################

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        if self.opt.pretrained!=None:
            self.opt.pretrained=os.path.join("./pretrained_ckpt",self.opt.pretrained)
        self.args = vars(self.opt)
        
        '------------ Load nnUNet planning -------------'
        planner=nnUNETPlanning(self.opt)
        CurrentPlan=planner.load_my_plans()
        #take always FULLRES
        self.opt.stage=len(CurrentPlan['plans_per_stage'])-1
        self.opt.planning_stage='nnUNetData_plans_v2.1_stage'+str(self.opt.stage)
           
        if self.opt.imageSize!=0:
            pass
        else:
            self.opt.imageSize=CurrentPlan['plans_per_stage'][self.opt.stage]['patch_size']
        '-------------'

        print('------------ Options -------------')
        for k, v in sorted(self.args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')


        # save to the disk
        if self.opt.name is None:
            self.opt.name=self.opt.encoder+'F'+str(self.opt.fold)
            
        self.opt.imageSize=[int(self.opt.imageSize[i]) for i in range(len(self.opt.imageSize))]
        self.opt.filters_Encoder=tuple([int(self.opt.filters_Encoder[i]) for i in range(len(self.opt.filters_Encoder))])
        if self.opt.region[0]!='None' and self.opt.dataroot!='Task001_BraTS2021':
            self.opt.region=tuple([tuple([int(i) for i in x.split(',')]) if len(x)>1 else (int(x),) for x in self.opt.region])
       
        expr_dir = os.path.join(self.opt.checkpoints_dir,self.opt.encoder,self.opt.name)
        out_dir = os.path.join(self.opt.output_dir, self.opt.name)
        util.mkdirs(expr_dir)
        util.mkdirs(out_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(self.args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        self.opt.expr_dir=expr_dir
        self.opt.out_dir=out_dir
        root_dir=self.opt.expr_dir
        max_epochs = self.opt.epochs
        val_interval = self.opt.val_interval
        Plots=VisualPlots(self.opt) #I will use to save some segmentation results. At the moment is only for plot loss curve
        
        return self.opt,root_dir,max_epochs,val_interval,Plots
