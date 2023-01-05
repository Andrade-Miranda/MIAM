#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:28:59 2022

@author: gustavo
"""

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from fvcore.nn import ActivationCountAnalysis


import argparse
from batchgenerators.utilities.file_and_folder_operations import join

from models.models import create_model
import torch

from util.testing_setup import load_trainingSetup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="pre-processing nnUNet test set",default='/home/gustavo/Data/results', required=False)
    parser.add_argument('-o', "--output_folder",default='flops', required=False, help="folder for saving flops")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default='Task004_Hektor', required=False)#'Task001_BraTS2021' 'Task004_Hektor'
    parser.add_argument('-m', '--model', help='models name', default="UNETR", required=False)
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")
        
    parser.add_argument("--GPU", dest='GPU', action='store_true',default=False, help=
    "test in GPU or CPU")
    
    #incluir una opcion para path de checkpoint
    args = parser.parse_args()
 
    #to solve: temporary solution to include Vitm0 and MCNN+VITm0
    if args.model[-1]=='0':
        model = args.model[:-1]
    else:
        model = args.model
        
    modelname=model+'F'+str(0)# only read fold 0
    
    if args.task_name.split('_')[-1]=='Hektor':
        task='Hektor2021'
    elif args.task_name.split('_')[-1]=='BraTS2021':
        task='BratS2021'
    
    args.chkdir=task+'/checkpoints/multicenter'
    
    chk_folder = join(args.input_folder,args.chkdir,model,modelname)
    args.checkpoints_dir=chk_folder
    args.folds=0
    output_folder = join('./Output',args.output_folder,model)
    args.output_dir=output_folder
    args.chkname='BestCHK.pth' # use this to ensure compatibility with load_trainingSetup function
    
    args.mode='MeanEnsemb'# use this mode to ensure compatibility with load_trainingSetup function
    opt= load_trainingSetup(join(chk_folder,'opt.txt'),args,args.folds)

    #to solve: temporary solution to include Vitm0 and MCNN+VITm0    
    if args.model[-1]=='0':
        opt.encoder=args.model
        
    model = create_model(opt)
    x=torch.rand((2,2,128,128,128))
    #y=model(x)
    
    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops))
    flops.total()
    
    act=ActivationCountAnalysis(model,x)
    print(act)

        


        


if __name__ == "__main__":
    main()





