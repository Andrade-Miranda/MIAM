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


from util.testing_setup import load_trainingSetup,Mode_NCrossval,Mode_MeanEnsemb,Mode_MeanEnsembBrats



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="pre-processing nnUNet test set",default='./nnUNet/data/nnUnet_raw/nnUNet_raw_data', required=False)
    parser.add_argument('-o', "--output_folder",default='predictions', required=False, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default='Task003_Hektor', required=False)
    parser.add_argument('-m', '--model', help='models name', default="CNN_h+VIT_n", required=False)
    parser.add_argument('-f', '--folds', nargs='+', default=None,
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")

    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")
    
    parser.add_argument("--mode", default='Nfold', help="modes to use: Nfold(separate), emsembling, Test time augmentation")
    
    parser.add_argument("--GPU", dest='GPU', action='store_true',default=False, help=
    "test in GPU or CPU")
    
    parser.add_argument('--chkname',
                        help='checkpoint extension, only available for one chk for folder',
                        required=False,
                        default='lastestCHK.pth')
    parser.add_argument('--chkdir',
                        help='checkpoint directory',
                        required=False,
                        default='./checkpoints')

    #incluir una opcion para path de checkpoint
    args = parser.parse_args()
    #task_name = args.task_name# a borrar
    #input_folder = join(args.input_folder,task_name,'imagesTs')# a borrar
    #num_threads_preprocessing = args.num_threads_preprocessing
    #num_threads_nifti_save = args.num_threads_nifti_save
    model = args.model
    mode=args.mode
    
    modelname=[args.model+'F'+str(args.folds[i]) for i in range(len(args.folds))]
    chk_folder = [join(args.chkdir,model,modelname[i]) for i in range(len(args.folds))]
    args.checkpoints_dir=chk_folder
    args.folds=[int(args.folds[i]) for i in range(len(args.folds))]
    if mode=='Nfold':
        output_folder = [join('./Output',args.output_folder,model,modelname[i]) for i in range(len(args.folds))]
    else:
        output_folder=join('./Output',args.output_folder,model+'_Ensemble')
    args.output_dir=output_folder

    opt= [load_trainingSetup(join(chk_folder[i],'opt.txt'),args,i) for i in range(len(args.folds))]
    [print("using model stored in ", chk_folder[i]) for i in range(len(args.folds))]
   
    
    if mode=='Nfold':
        Mode_NCrossval(args,opt,output_folder)
    elif mode=='MeanEnsemb':
        if args.task_name.split('_')[-1]=='BraTS2021':
            Mode_MeanEnsembBrats(args,opt)
        else:
            Mode_MeanEnsemb(args,opt)
   # elif mode=='TTA':
        #Test_time_Augmentation(opt,fold=0)

        


        


if __name__ == "__main__":
    main()
