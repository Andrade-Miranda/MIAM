#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:41:01 2022

@author: gustavo
"""

import torch
import numpy as np
from config.train_setup import TrainSetup
from models.models import create_model
from copy import deepcopy
from argparse import Namespace
from util.metrics import metricStatistics

from data.data_loader import CreateDataLoader
from monai.data import (
    decollate_batch,
    TestTimeAugmentation
)
from batchgenerators.utilities.file_and_folder_operations import join,maybe_mkdir_p,subfiles,isfile
import nibabel as nib
import os
from functools import partial

from monai.transforms import (
    Compose,
    RandSpatialCropd,
    RandFlipd,
    EnsureTyped,
    AsDiscrete
)



def predict_from_folder(opt=None):
    """
        here we use the standard naming scheme to generate list_of_lists and output_files needed by predict_cases

    :param model:
    :param input_folder:
    :param output_folder:
    :param folds:
    :param save_npz:
    :param num_threads_preprocessing:
    :param num_threads_nifti_save:
    :param lowres_segmentations:
    :param part_id:
    :param num_parts:
    :param tta:
    :param mixed_precision:
    :param overwrite_existing: if not None then it will be overwritten with whatever is in there. None is default (no overwrite)
    :return:
    """

    print("loading parameters for folds,", opt.fold)
    trainer= restore_Model(join(opt.checkpoints_dir,opt.checkpoint),opt)


    return trainer



def restore_Model(file,opt):
    
    model = create_model(opt)
    pkl_file=file
    model.load_state_dict(
    torch.load(pkl_file,map_location=opt.device)
    )
    return model

def load_trainingSetup(file_name,args,numiter):
    lista=[]
    with open(file_name, 'rb') as opt_file:
        lines = opt_file.readlines()
        for line in lines[2:-1]:
            a=line.decode("utf-8").split(':')
            value=a[1][1:-1]
            key=a[0]
            
            if key=="gpu_ids":# cambio para que funcione con CPU
                if not args.GPU:
                   value=[]
                else:
                   value=[int(value[1:-1])]
            elif key=='dataroot':
                value=args.task_name
            elif key=='output_dir':
                value=args.output_dir[numiter]
            elif key=='checkpoints_dir':
                value=args.checkpoints_dir[numiter]
            elif key=="device":
                if not args.GPU:
                   value='cpu'
            elif key=='dataset_mode':
                value=args.mode
            elif key=='TrainConfig':
                value='TestConfig'
            elif value[0].isnumeric() and len(value)>1:
                if value[1]=='.':
                    value=float(value)
                else:
                    value=int(value)
            elif value[0].isnumeric():
                value=int(value)
            elif value[0]=='[' and value[2:-2]!='None':
                value=list([int(i[1:]) for i in value[:-1].split(',')])
            elif value[0]=='[' and value[2:-2]=='None':
                value='None'
            elif value[0]=='(':
                if value[1]=='(':
                    value=((1, 4), (1, 4, 2), (4,))
                else:
                    value=tuple([int(i[1:]) for i in value[:-1].split(',')]) 
            elif value=='True':
                value=True
            elif value=='False':
                value=False
               
            lista.append((key,value))
        lista.append(('checkpoint',args.chkname))
        opt=dict(lista)
        if opt['encoder'] in ['VIT_n','VIT_s','VIT_m','MVIT_n','MVIT_s','MVIT_m','CNN+VIT2Stream','SegResNetVAE','SegResNet','UNETR','Unet','SwinTrans3D']:
            opt['hybrid']=False
        else:
            opt['hybrid']=True
        # for k, v in opt.items():
        #     pars.add_argument('--' + k, default=v)
        
        return Namespace(**opt)


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids

def Mode_NCrossval(args,opt,output_folder):
    for i in range(len(args.folds)):
        model=predict_from_folder(opt[i])
    
        data_loader = CreateDataLoader(opt[i])
        testConfig=TrainSetup(opt[i],model)
        test_loader = data_loader.load_test()
    
        maybe_mkdir_p(opt[i].output_dir)
        metric_values_tumor = [] #DICE
        HDistance = []
        AVgSurfDis = []
        recall_values_tumor = []
        precision_values_tumor = []
        results=[]
        model.eval()
        with torch.no_grad():#Context-manager that disabled gradient calculation.
            for batchIt in range(80):#range(len(data_loader)):
                val_data = next(test_loader)
                val_inputs,val_labels= (
                         val_data["image"].to(opt[i].device),
                         val_data["label"].to(opt[i].device))
                val_outputs = testConfig.Config.inference(val_inputs)
                val_outputs = [testConfig.Config.post_trans(i) for i in decollate_batch(val_outputs)]
                testConfig.Config.dice_metric(y_pred=val_outputs, y=val_labels)
                testConfig.Config.Recall_Precision(y_pred=val_outputs, y=val_labels)
                testConfig.Config.HausdorffDis(y_pred=val_outputs, y=val_labels)
                testConfig.Config.SurfDis(y_pred=val_outputs, y=val_labels)

                metric = testConfig.Config.dice_metric.aggregate().item()
                metric_values_tumor.append(metric)
            
                HD = testConfig.Config.HausdorffDis.aggregate().item()
                HDistance.append(HD)
            
                SurfDis = testConfig.Config.SurfDis.aggregate().item()
                AVgSurfDis.append(SurfDis)

                Recall_Precision = testConfig.Config.Recall_Precision.aggregate()
                recall=Recall_Precision[0].item()
                precision=Recall_Precision[1].item()
                recall_values_tumor.append(recall)
                precision_values_tumor.append(precision)
                
                testConfig.Config.dice_metric.reset()
                testConfig.Config.Recall_Precision.reset()
                testConfig.Config.HausdorffDis.reset()
                testConfig.Config.SurfDis.reset()

                
                if not os.path.exists(join('./Output/predictions/',opt[i].dataroot)):#i==0:#check si existe true labels
                    ### INPUT IMAGE
                    maybe_mkdir_p(join('./Output/predictions/',opt[i].dataroot))
                    inp=np.moveaxis(val_inputs[0,:,:,:,:].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                    new_image = nib.Nifti1Image(inp, affine=np.eye(4))
                    new_image.header.get_xyzt_units()
                    new_image.to_filename(join('./Output/predictions/',opt[i].dataroot,val_data['keys'][0]+'.nii.gz'))
                
                    ####LABEL True
                    maybe_mkdir_p(join('./Output/predictions/',opt[i].dataroot))
                    label=np.moveaxis(val_labels[0,:,:,:,:].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                    new_label = nib.Nifti1Image(label, affine=np.eye(4))
                    new_label.header.get_xyzt_units()
                    new_label.to_filename(join('./Output/predictions/',opt[i].dataroot,val_data['keys'][0]+'_labelTrue'+'.nii.gz')) 
            
                ####LABEL predictions
                label=np.moveaxis(val_outputs[0].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
                new_label = nib.Nifti1Image(label, affine=np.eye(4))
                new_label.header.get_xyzt_units()
                new_label.to_filename(join(output_folder[i],val_data['keys'][0]+'_labelPred'+'.nii.gz'))  
            
                print(
                    f"Patients: {val_data['keys'][0]} "
                    f" Dice: {metric:.5f}"
                    f" ASD: {SurfDis:.5f} "
                    f" HD: {HD:.5f} "
                    f" Recall: {recall:.5f} "
                    f" Precision: {precision:.5f} "
                    ,flush=True
                    )
                results.append((val_data['keys'][0],metric,SurfDis,HD,recall,precision))
            results.sort()
        Dsc_ST,ASD_ST,HD_ST,recall_ST,precision_ST=metricStatistics(metric_values_tumor,AVgSurfDis,HDistance,recall_values_tumor,precision_values_tumor)
        results.append(('Average',Dsc_ST,ASD_ST,HD_ST,recall_ST,precision_ST))
        savePredictions(output_folder[i],results)
        results=[]                

def Mode_MeanEnsemb(args,opt):
    LastTransf=AsDiscrete(threshold_values=True)
    output_folder = join('./Output',args.output_folder,opt[0].encoder+'_Ensemble')
    data_loader = CreateDataLoader(opt[args.folds[0]])
    test_loader = data_loader.load_test()#as is the same model use same pre-processing
    models=[predict_from_folder(opt[i]).eval() for i in range(len(args.folds))]
    testConfig=[TrainSetup(opt[args.folds[i]],models[i]) for i in range(len(args.folds))]
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    results=[]
    val_outStack=[]
    with torch.no_grad():#Context-manager that disabled gradient calculation.
        for batchIt in range(len(data_loader)):
            val_data = next(test_loader)
            val_inputs,val_labels= (
                        val_data["image"].to(opt[0].device),
                        val_data["label"].to(opt[0].device))
            val_outputs = [testConfig[i].Config.inference(val_inputs) for i in range(len(args.folds))]
            for j in range(len(val_outputs)):
                val_outStack.append([testConfig[j].Config.post_trans(i) for i in decollate_batch(val_outputs[j])][0])
            val_outputs = LastTransf(torch.stack(val_outStack, dim=0).mean(dim=0))[None,:]
            testConfig[0].Config.dice_metric(y_pred=val_outputs, y=val_labels)
            testConfig[0].Config.dice_metric_batch(y_pred=val_outputs, y=val_labels)
                                
            metric = testConfig[0].Config.dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = testConfig[0].Config.dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            testConfig[0].Config.dice_metric.reset()
            testConfig[0].Config.dice_metric_batch.reset()       
            val_outStack=[]
            ####LABEL predictions
            maybe_mkdir_p(output_folder)
            label=np.moveaxis(val_outputs[0].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
            labelMC=np.zeros((label.shape[0],label.shape[1],label.shape[2]))
            for j in [1,0,2]:
                labelMC[label[:,:,:,j]==1]=j+1 
            new_label = nib.Nifti1Image(labelMC, affine=np.eye(4))
            new_label.header.get_xyzt_units()
            new_label.to_filename(join(output_folder,val_data['keys'][0]+'_labelPred'+'.nii.gz'))  
            print(
                    f"Patients: {val_data['keys'][0]} "
                    f"current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    ,flush=True
                )
            results.append((val_data['keys'][0],metric,metric_tc,metric_wt,metric_et))
        results.sort()
    savePredictions(output_folder,results)
        
def Test_time_Augmentation(opt,fold=0):
    output_folder = opt[fold].output_dir+'_TTA'
    data_loader = CreateDataLoader(opt[fold])
    test_loader = data_loader.load_test()#as is the same model use same pre-processing
    model=predict_from_folder(opt[fold]).eval()
    testConfig=TrainSetup(opt[fold],model) 
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    results=[]
    
    tt_aug = TestTimeAugmentation(
                    transformations(),
                    batch_size=opt[fold].Val_batchSize,
                    num_workers=0,
                    inferrer_fn=partial(infer_seg,models=model,opt=opt,fold=fold,testConfig=testConfig),  # fn to infer segmentation
                    device=opt[fold].device
                    )

    with torch.no_grad():#Context-manager that disabled gradient calculation.
        for batchIt in range(len(data_loader)):
            val_data = next(test_loader)
            val_inputs={'image': val_data["image"],
                        'label': val_data["label"]}
            

            mode_tta, mean_tta, std_tta, vvc_tta = tt_aug(val_inputs, num_examples=5)
            
            val_outputs=mean_tta
            testConfig.Config.dice_metric(y_pred=val_outputs, y=val_inputs['label'])
            testConfig.Config.dice_metric_batch(y_pred=val_outputs, y=val_inputs['label'])
                                
            metric = testConfig.Config.dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = testConfig.Config.dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            testConfig.Config.dice_metric.reset()
            testConfig.Config.dice_metric_batch.reset()       
        
            ####LABEL predictions
            maybe_mkdir_p(output_folder)
            label=np.moveaxis(val_outputs[0].cpu().detach().numpy(),(0,1,2),(-1,-2,-3))
            labelMC=np.zeros((label.shape[0],label.shape[1],label.shape[2]))
            for j in [1,0,2]:
                labelMC[label[:,:,:,j]==1]=j+1 
            new_label = nib.Nifti1Image(labelMC, affine=np.eye(4))
            new_label.header.get_xyzt_units()
            new_label.to_filename(join(output_folder,val_data['keys'][0]+'_labelPred'+'.nii.gz'))  
            print(
                    f"Patients: {val_data['keys'][0]} "
                    f"current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    ,flush=True
                )
            results.append((val_data['keys'][0],metric,metric_tc,metric_wt,metric_et))
        results.sort()
    savePredictions(output_folder,results)    
    

def savePredictions(output_folder,results):
    file_name = os.path.join(output_folder, 'predictions.txt')
    with open(file_name, 'wt') as pred_file:
        pred_file.write('------------ Options -------------\n')
        for patients,dice,SurfDis,HD,recall,precision in results:
            if patients!="Average":
                pred_file.write(f"Patients: {patients} "
                                f"DICE: {dice:.5f}"
                                f" ASD: {SurfDis:.5f} HD: {HD:.4f} Recall: {recall:.5f} Precision: {precision:.5f} \n")
            else:
                pred_file.write(f"Metrics {patients}: "
                                f"DICE: {dice[0]:.5f}"u"\u00B1"f"{dice[1]:.5f} "
                                f"ASD: {SurfDis[0]:.5f}"u"\u00B1"f"{SurfDis[1]:.5f} "
                                f"HD: {HD[0]:.5f}"u"\u00B1"f"{HD[1]:.5f} "
                                f"Recall: {recall[0]:.5f}"u"\u00B1"f"{recall[1]:.5f} "
                                f"Precision: {precision[0]:.5f}"u"\u00B1"f"{precision[1]:.5f} \n")
        pred_file.write('-------------- End ----------------\n')

def infer_seg(val_inputs, models, opt,fold,testConfig):
    val_outputs=testConfig.Config.inference(val_inputs[0])
    val_outputs = [testConfig.Config.post_trans(i) for i in decollate_batch(val_outputs)]
    return val_outputs[0][None,:]


def transformations():
        tta_transforms = Compose(
    [
        RandSpatialCropd(keys=["image", "label"], roi_size=[128,128,128], random_size=True),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        #RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        #RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
)
        return tta_transforms

