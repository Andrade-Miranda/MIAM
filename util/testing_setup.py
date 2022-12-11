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


import SimpleITK as sitk
from nnUNet.nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from batchgenerators.utilities.file_and_folder_operations import *



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
    checkpoint = torch.load(file,map_location=opt.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['model_state_dict'],strict=False)
    else:
        model.load_state_dict(
            checkpoint,strict=False)
    print("Replace Default initialization... LOAD TRAINED WEIGHTS")
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
                if args.mode=='MeanEnsemb':
                    value=args.output_dir
                else:
                    value=args.output_dir[numiter]
            elif key=='checkpoints_dir':
                value=args.checkpoints_dir[numiter]
            elif key=='pretrained':
                 if value=='None':
                     value=None
            elif key=='isTrain':
                value=False
            elif key=="device":
                if not args.GPU:
                   value='cpu'
            elif key=='dataset_mode':
                value=args.mode
            elif key=='yh_run_model':
                value='test'
            elif key=='TrainConfig':
                if args.task_name=='Task004_BraTS2021':
                    value='Test_ConfigBrats'
                else:
                    value='TestConfig'
            elif key=='conv_kernel_sizes':
                newValue=value.translate({ord(i): None for i in '[,] '})
                value= [(list(map(int,newValue[x:x+3]))) for x in range(0, len(newValue), 3)]
            elif key=='pool_op_kernel_sizes':
                newValue=value.translate({ord(i): None for i in '[,] '})
                value= [(list(map(int,newValue[x:x+3]))) for x in range(0, len(newValue), 3)]
            elif key=='num_pool_per_axis':
                newValue=value.translate({ord(i): None for i in '[,] '})                
                value= [int(x) for x in newValue]    
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
        
        if args.task_name.split('_')[-1]=='BraTS2021':
            lista.append(('region_class_order',(2,1,4)))
            lista.append(('isbrats',True))
        else:
            lista.append(('region_class_order',None))
            lista.append(('isbrats',False))
            
        opt=dict(lista)
        if opt['encoder'] in ['VIT_n','VIT_s','VIT_m','MVIT_n','MVIT_s','MVIT_m','CNN+VIT2Stream','SegResNetVAE','SegResNet','UNETR','Unet','SwinTrans3D','MCNN_h']:
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

def Mode_NCrossval(args,opt,output_folder):#### need to be updated
    for i in range(len(args.folds)):
        maybe_mkdir_p(opt[i].output_dir)
        model=predict_from_folder(opt[i]).eval() 
        data_loader = CreateDataLoader(opt[i])
        testConfig=TrainSetup(opt[i],model)
        test_loader = data_loader.load_test()
        with torch.no_grad():#Context-manager that disabled gradient calculation.
            for batchIt in range(len(data_loader)):
                val_data = next(test_loader)
                val_inputs,properties_dict= (
                        val_data["image"].to(opt[0].device),
                        val_data["properties"])
                val_outputs = testConfig.Config.inference(val_inputs)
                val_outputs = testConfig.Config.post_trans(val_outputs[0])
          
                out_fname=join(output_folder[i],val_data['keys'][0]+'.nii.gz')
                save_segmentation_nifti_from_softmax(val_outputs, out_fname,
                                         properties_dict, order=1,
                                         region_class_order= opt[i].region_class_order,
                                         seg_postprogess_fn= None, seg_postprocess_args= None,
                                         resampled_npz_fname= None,
                                         non_postprocessed_fname= None, force_separate_z= None,
                                         interpolation_order_z= 0, verbose= True,isbrats=opt[i].isbrats)
           
                del val_outputs
                del val_data
                torch.cuda.empty_cache()             


def Mode_MeanEnsembBrats(args,opt):#we don't applied argmax or discrete give directly the sigmoid, region_class_order 
    output_folder = args.output_dir
    maybe_mkdir_p(output_folder)
    data_loader = CreateDataLoader(opt[args.folds[0]])
    test_loader = data_loader.load_test()#as is the same model use same pre-processing
    models=[predict_from_folder(opt[i]).eval() for i in range(len(args.folds))]
    testConfig=[TrainSetup(opt[i],models[i]) for i in range(len(args.folds))]
    with torch.no_grad():#Context-manager that disabled gradient calculation.
        for batchIt in range(len(data_loader)):
            val_data = next(test_loader)
            val_inputs,properties_dict= (
                        val_data["image"].to(opt[0].device),
                        val_data["properties"])
            val_outputs = [testConfig[i].Config.inference(val_inputs) for i in range(len(args.folds))]
            val_outStack=[]
            for j in range(len(val_outputs)):
                val_outStack.append([testConfig[j].Config.post_trans(i) for i in decollate_batch(val_outputs[j])][0])
            val_outputs = torch.stack(val_outStack, dim=0).mean(dim=0)
            out_fname=join(output_folder,val_data['keys'][0]+'.nii.gz')

            save_segmentation_nifti_from_softmax(val_outputs, out_fname,
                                                 properties_dict, order=1, 
                                                 region_class_order=(2,1,4),
                                                 seg_postprogess_fn= None, seg_postprocess_args= None,
                                                 resampled_npz_fname= None,
                                                 non_postprocessed_fname= None, force_separate_z= None,
                                                 interpolation_order_z= 0, verbose= True,isbrats=True)
            del val_outputs
            del val_data
            torch.cuda.empty_cache()


def Mode_MeanEnsemb(args,opt):
    output_folder = args.output_dir
    maybe_mkdir_p(output_folder)
    data_loader = CreateDataLoader(opt[args.folds[0]])
    test_loader = data_loader.load_test()#as is the same model use same pre-processing
    models=[predict_from_folder(opt[i]).eval() for i in range(len(args.folds))]
    testConfig=[TrainSetup(opt[i],models[i]) for i in range(len(args.folds))]
    with torch.no_grad():#Context-manager that disabled gradient calculation.
        for batchIt in range(len(data_loader)):
            val_data = next(test_loader)
            val_inputs,properties_dict= (
                        val_data["image"].to(opt[0].device),
                        val_data["properties"])
            val_outputs = [testConfig[i].Config.inference(val_inputs) for i in range(len(args.folds))]
            val_outStack=[]
            for j in range(len(val_outputs)):
                val_outStack.append([testConfig[j].Config.post_trans(i) for i in decollate_batch(val_outputs[j])][0])
            val_outputs = testConfig[0].Config.postLast(torch.stack(val_outStack, dim=0).mean(dim=0))
          
            out_fname=join(output_folder,val_data['keys'][0]+'.nii.gz')
            save_segmentation_nifti_from_softmax(val_outputs, out_fname,
                                         properties_dict, order=1,
                                         region_class_order= None,
                                         seg_postprogess_fn= None, seg_postprocess_args= None,
                                         resampled_npz_fname= None,
                                         non_postprocessed_fname= None, force_separate_z= None,
                                         interpolation_order_z= 0, verbose= True)
           
            del val_outputs
            del val_data
            torch.cuda.empty_cache()
            

def save_segmentation_nifti_from_softmax(segmentation_softmax, out_fname,
                                         properties_dict, order=1,
                                         region_class_order= None,
                                         seg_postprogess_fn= None, seg_postprocess_args= None,
                                         resampled_npz_fname= None,
                                         non_postprocessed_fname= None, force_separate_z= None,
                                         interpolation_order_z= 0, verbose= True, isbrats=False):
    """
    This is a utility for writing segmentations to nifto and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifto export
    There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        segmentation_softmax = np.load(segmentation_softmax)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict[0].get('size_after_cropping')
    shape_original_before_cropping = properties_dict[0].get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict[0].get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict[0].get('original_spacing'))
            elif get_do_separate_z(properties_dict[0].get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict[0].get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict[0].get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax.detach().cpu().numpy(), shape_original_after_cropping, is_seg=True,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing[0].detach().cpu().numpy()# i did ya argmax
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        if isbrats:
            nclass=[1,0,2]
            for i, c in enumerate(region_class_order):
                seg_old_spacing_final[seg_old_spacing.detach().cpu().numpy()[nclass[i]] > 0.5] = c
                
            seg_old_spacing = seg_old_spacing_final 
        #else:
            #for i, c in enumerate(region_class_order):
             #   seg_old_spacing_final[seg_old_spacing.detach().cpu().numpy()[i] > 0.5] = c
            #seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict[0].get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict[0]['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict[0]['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict[0]['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)



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
