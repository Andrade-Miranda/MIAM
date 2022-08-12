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


from util.testing_setup import predict_from_folder,savePredictions,load_trainingSetup


import torch
import numpy as np
from config.train_setup import TrainSetup

from data.data_loader import CreateDataLoader
from monai.data import (
    decollate_batch,
)
from batchgenerators.utilities.file_and_folder_operations import join,maybe_mkdir_p,subfiles,isfile
import nibabel as nib






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                     " order (same as training). Files must be named "
                                                     "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                     "identifier (0000, 0001, etc)",default='./nnUNet/data/nnUnet_raw/nnUNet_raw_data', required=False)
    parser.add_argument('-o', "--output_folder",default='predictions', required=False, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.',
                        default='Task001_BraTS2021', required=False)
    parser.add_argument('-m', '--model', help='models name', default=["MCNN_h+VIT_n",'UNETR'], required=False)
    parser.add_argument('-f', '--folds', default=0,
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

    args = parser.parse_args()
    #task_name = args.task_name# a borrar
    #input_folder = join(args.input_folder,task_name,'imagesTs')# a borrar
    #num_threads_preprocessing = args.num_threads_preprocessing
    #num_threads_nifti_save = args.num_threads_nifti_save
    models = args.model
    mode=args.mode
    
    modelname=[model+'F'+str(args.folds) for model in models]
    output_folder = [join('./Output',args.output_folder,modelname[i]) for i in range(len(models))]
    chk_folder = [join('./checkpoints',models[i],modelname[i]) for i in range(len(models))]
    args.output_dir=output_folder
    args.checkpoints_dir=chk_folder
    args.folds=[int(args.folds[i]) for i in range(len(args.folds))]

    opt= [load_trainingSetup(join(chk_folder[i],'opt.txt'),args,i) for i in range(len(models))]
    [print("using model stored in ", chk_folder[i]) for i in range(len(args.folds))]
   
    
 

    if mode=='Nfold':
        for i in range(len(models)):
            model1=predict_from_folder(opt[0])
            model2=predict_from_folder(opt[1])
            
            data_loader = CreateDataLoader(opt[0])
            testConfig1=TrainSetup(opt[0],model1)
            testConfig2=TrainSetup(opt[1],model2)
            test_loader = data_loader.load_test()
        
            maybe_mkdir_p(opt[0].output_dir)
            maybe_mkdir_p(opt[1].output_dir)
            metric_values1 = []
            metric_values_tc1 = []
            metric_values_wt1 = []
            metric_values_et1 = []
            
            metric_values2 = []
            metric_values_tc2 = []
            metric_values_wt2 = []
            metric_values_et2 = []
            
            results1=[]
            results2=[]
            model1.eval()
            model2.eval()
            with torch.no_grad():#Context-manager that disabled gradient calculation.
                for batchIt in range(len(data_loader)):
                    val_data = next(test_loader)
                    val_inputs,val_labels= (
                             val_data["image"].to(opt[0].device),
                             val_data["label"].to(opt[0].device))
                    val_outputs1 = testConfig1.Config.inference(val_inputs)
                    val_outputs1 = [testConfig1.Config.post_trans(i) for i in decollate_batch(val_outputs1)]
                    testConfig1.Config.dice_metric(y_pred=val_outputs1, y=val_labels)
                    testConfig1.Config.dice_metric_batch(y_pred=val_outputs1, y=val_labels)
                                    
                    metric1 = testConfig1.Config.dice_metric.aggregate().item()
                    metric_values1.append(metric1)
                    metric_batch1 = testConfig1.Config.dice_metric_batch.aggregate()
                    metric_tc1 = metric_batch1[0].item()
                    metric_values_tc1.append(metric_tc1)
                    metric_wt1 = metric_batch1[1].item()
                    metric_values_wt1.append(metric_wt1)
                    metric_et1 = metric_batch1[2].item()
                    metric_values_et1.append(metric_et1)
                    testConfig1.Config.dice_metric.reset()
                    testConfig1.Config.dice_metric_batch.reset()
                    
                    val_outputs2 = testConfig2.Config.inference(val_inputs)
                    val_outputs2 = [testConfig2.Config.post_trans(i) for i in decollate_batch(val_outputs2)]
                    testConfig2.Config.dice_metric(y_pred=val_outputs2, y=val_labels)
                    testConfig2.Config.dice_metric_batch(y_pred=val_outputs2, y=val_labels)
                                    
                    metric2 = testConfig2.Config.dice_metric.aggregate().item()
                    metric_values2.append(metric2)
                    metric_batch2 = testConfig2.Config.dice_metric_batch.aggregate()
                    metric_tc2 = metric_batch2[0].item()
                    metric_values_tc2.append(metric_tc2)
                    metric_wt2 = metric_batch2[1].item()
                    metric_values_wt2.append(metric_wt2)
                    metric_et2 = metric_batch2[2].item()
                    metric_values_et2.append(metric_et2)
                    testConfig2.Config.dice_metric.reset()
                    testConfig2.Config.dice_metric_batch.reset()
                    
                    
                    ### INPUT IMAGE
                    maybe_mkdir_p(join('./Output/predictions/',opt[0].dataroot))
                    inp=np.moveaxis(val_inputs[0,:,:,:,:].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                    new_image = nib.Nifti1Image(inp, affine=np.eye(4))
                    new_image.header.get_xyzt_units()
                    new_image.to_filename(join('./Output/predictions/',opt[0].dataroot,val_data['keys'][0]+'.nii.gz'))
                    
                    ####LABEL True
                    maybe_mkdir_p(join('./Output/predictions/',opt[0].dataroot))
                    label=np.moveaxis(val_labels[0,:,:,:,:].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                    labelTrue=np.zeros((label.shape[0],label.shape[1],label.shape[2]))
                    for j in [1,0,2]:
                        labelTrue[label[:,:,:,j]==1]=j+1 
                    new_label = nib.Nifti1Image(labelTrue, affine=np.eye(4))
                    new_label.header.get_xyzt_units()
                    new_label.to_filename(join('./Output/predictions/',opt[0].dataroot,val_data['keys'][0]+'_labelTrue'+'.nii.gz')) 
                
                    ####LABEL predictions MCNN+VIT
                    label=np.moveaxis(val_outputs1[0].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                    labelMC=np.zeros((label.shape[0],label.shape[1],label.shape[2]))
                    for j in [1,0,2]:
                        labelMC[label[:,:,:,j]==1]=j+1 
                    new_label = nib.Nifti1Image(labelMC, affine=np.eye(4))
                    new_label.header.get_xyzt_units()
                    new_label.to_filename(join(output_folder[0],val_data['keys'][0]+'_labelPred'+'.nii.gz'))  
                
                    print(
                        f"Patients: {val_data['keys'][0]} "
                        f"current mean dice: {metric1:.4f}"
                        f" tc: {metric_tc1:.4f} wt: {metric_wt1:.4f} et: {metric_et1:.4f}"
                        ,flush=True
                        )
                    results1.append((val_data['keys'][0],metric1,metric_tc1,metric_wt1,metric_et1))
                    
                    
                    ####LABEL predictions UNTR
                    label=np.moveaxis(val_outputs2[0].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                    labelMC=np.zeros((label.shape[0],label.shape[1],label.shape[2]))
                    for j in [1,0,2]:
                        labelMC[label[:,:,:,j]==1]=j+1 
                    new_label = nib.Nifti1Image(labelMC, affine=np.eye(4))
                    new_label.header.get_xyzt_units()
                    new_label.to_filename(join(output_folder[1],val_data['keys'][0]+'_labelPred'+'.nii.gz'))  
                
                    print(
                        f"Patients: {val_data['keys'][0]} "
                        f"current mean dice: {metric2:.4f}"
                        f" tc: {metric_tc2:.4f} wt: {metric_wt2:.4f} et: {metric_et2:.4f}"
                        ,flush=True
                        )
                    results2.append((val_data['keys'][0],metric2,metric_tc2,metric_wt2,metric_et2))
                    
                    
                    
                results1.sort()
                results2.sort()
            savePredictions(output_folder[0],results1)
            savePredictions(output_folder[1],results2)

        



if __name__ == "__main__":
   main()
