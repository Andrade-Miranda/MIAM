#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
from batchgenerators.utilities.file_and_folder_operations import join


from util.testing_setup import load_trainingSetup,Mode_NCrossval,Mode_MeanEnsemb,Mode_MeanEnsembBrats,Mode_MCdropout
import os



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_fold_prediction', help="pre-processing nnUNet test set",default='./nnUNet/data/nnUnet_raw/nnUNet_raw_data', required=False)
    parser.add_argument('-y_True', "--y_true_dir",default=None, required=False, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default='Task003_Hektor', required=False)
    parser.add_argument('-m', '--model', help='models name', default="CNN_h+VIT_n", required=False)
    parser.add_argument('-f', '--Nfolds', nargs='+', default=None,
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")
    
    parser.add_argument("--mode", default='Nfold', help="modes to use: Nfold(separate), emsembling, Test time augmentation")
    parser.add_argument("--MCDropOutRate", required=False, default=0.4, type=float, help="Only used if mode is MC Dropout")
    
    parser.add_argument("--GPU", dest='GPU', action='store_true',default=False, help=
    "test in GPU or CPU")

    parser.add_argument("--saveSoftmax", dest='saveSoftmax', action='store_true',default=False, help=
    "save probability prediction")
    parser.add_argument("--sigmoid", dest='sigmoid', action='store_true',default=False, help=
    "postprocessing using sigmoid(True) or softmax(False)")## in new version i will omit this option since it will be include in the training
    
    parser.add_argument("--postprocessing", dest='postprocessing', default=None, 
                        required=False, help="option for postprocessing data")
    
    parser.add_argument("--postpro_dir", dest='postpro_dir', default='/home/gustavo/Data/dataset/picai/dataset_test/picai_test/labels-WG', 
                        required=False, help="directory of mask used for postprocessing data")

    parser.add_argument('--chkname',
                        help='checkpoint extension, only available for one chk for folder',
                        required=False,
                        default='BestCHK.pth')
    
    parser.add_argument('--Only_enable_evaluation',action='store_true', dest='Only_enable_evaluation', default=False,
                    help="enable only evaluation setup")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb',action='store_true', dest='enable_wandb', default=False,
                    help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='PicaiTestSet', type=str,
                    help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', action='store_true',dest='wandb_ckpt', default=False,
                       help="Save model checkpoints as W&B Artifacts.")
    

    #realizar todos los setup##########################""""
    args = parser.parse_args()
    #checkpoints have to be saved in './checkpoints' folder. 
    #if you follow default MIAM training and test setup, checkpoints are already save it by default
    output_folderSoftmax='Softmax'# folder to save sofmax in case --saveSoftmax True
    output_fold_prediction='predictions' #folder to save prediction

    encoder = args.model.split('__')[0] # take only the model name (UNETR, Swin, etc)
    mode=args.mode

    modelname=[args.model+'F'+str(args.Nfolds[i]) for i in range(len(args.Nfolds))]
    args.checkpoints_dir = [join('./checkpoints',args.task_name,encoder,modelname[i]) for i in range(len(args.Nfolds))]
    args.folds=[int(args.Nfolds[i]) for i in range(len(args.Nfolds))]
    outputTestFolder=args.input_fold_prediction.split('/')[-2]#save results in a folder with the same name of the test dataset

    if mode=='Nfold':
        output_folder = [join('./nnUNet/data/nnUnet_raw/results',output_fold_prediction,outputTestFolder,encoder,modelname[i]) for i in range(len(args.folds))]
        output_folderSoftmax = [join('./nnUNet/data/nnUnet_raw/results',output_folderSoftmax,outputTestFolder,encoder,modelname[i]) for i in range(len(args.folds))]
    elif mode=='MeanEnsemb':
        output_folder=join('./nnUNet/data/nnUnet_raw/results',output_fold_prediction,outputTestFolder,encoder,args.model+'_Ensemble')
        output_folderSoftmax=join('./nnUNet/data/nnUnet_raw/results',output_folderSoftmax,outputTestFolder,encoder,args.model+'_Ensemble')
    elif mode=='MCdropOut':  
        output_folder=join('./nnUNet/data/nnUnet_raw/results',output_fold_prediction,outputTestFolder,encoder,args.model+'_MCDropOut-'+str(args.MCDropOutRate))
        output_folderSoftmax=join('./nnUNet/data/nnUnet_raw/results',output_folderSoftmax,outputTestFolder,encoder,args.model+'_MCDropOut-'+str(args.MCDropOutRate))


    args.output_pred_dir=output_folder
    args.outputSoft_dir=output_folderSoftmax

    opt= [load_trainingSetup(join(args.checkpoints_dir[i],'opt.txt'),args,i) for i in range(len(args.folds))]
    [print("using model stored in ", args.checkpoints_dir[i]) for i in range(len(args.folds))]
   
    
    if mode=='Nfold':
        Mode_NCrossval(opt,args.Nfolds)#multiple  folds 
    elif mode=='MeanEnsemb':
        if args.task_name.split('_')[-1]=='BraTS2021':
            Mode_MeanEnsembBrats(args,opt)
        else:
            Mode_MeanEnsemb(opt,args.Nfolds)
    elif mode=='MCdropOut':
        Mode_MCdropout(opt,args.Nfolds) #only available for ensembling mode

        


        


if __name__ == "__main__":
    main()
