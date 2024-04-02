# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:04:37 2021

@author: Wistan
"""


import os
import sys
import torch
import numpy as np
import nibabel as nib
import saliency.core as saliency

from models.Models import *
from utils.Tools import replace_layers
from skimage.transform import resize

#------------------------------------     PARAMETERS     ------------------------------------#


# Base paths
loading = './'


# List of folds
trials = [
            0,
            1,
            2,
          ]


# Fixed Parameters (since training is complete)
params = {
            'net_id': 11,
            'feature': 'CPS',
            'cutoff': 1,
            'batch': 1,
            'size': 192,
            'offsetx': 0,
            'offsety': 0,
            'offsetz': 0,
         }


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#-------------------------------     Save Function     ------------------------------------#



def save_nii(attr, extension, affine, output, trial_folder, filename, do_absolute):
    
    # Save attribution map as .nii
    save = output[:-1] + extension + output[-1] + trial_folder
    os.makedirs(save, exist_ok=True)
    save_file = save + 'Raw_attrs_IG' + extension + '_' + filename
    ni_img = nib.Nifti1Image(attr, affine=affine)
    nib.save(ni_img, save_file)
    
    # Save absolute map if needed
    if (do_absolute):
        attr_abs = np.abs(attr)
        save_abs = save.replace('Raw_attrs/', 'Raw_attrs(absolute)/')
        os.makedirs(save_abs, exist_ok=True)
        ni_img_abs = nib.Nifti1Image(attr_abs, affine=affine)
        nib.save(ni_img_abs, save_abs + 'Raw_attrs(absolute)_IG' + extension + '_' + filename)
#-------------------------------     call_model_function     --------------------------------#
def call_model_function(images,
                        call_model_args=None,
                        expected_keys=None):
    
    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    images.requires_grad = True
    
    target_class_idx = call_model_args['target']
    clf_features = call_model_args['clf_features']
    output = call_model_args['network'](images, clf_features)
    #target = output[0,target_class_idx]
    target = output[0][0, output[0].argmax()]
    call_model_args['network'].zero_grad()
    target.backward()
    
    grads = images.grad.data.squeeze(axis=1)
    gradients = grads.cpu().detach().numpy()
    
    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

#-------------------------------     INTEGRATED GRADIENTS     --------------------------------#
def IG(loader, checkpoints_path , output, do_absolute, args):
    
    # For each trial (=fold)
    for c, checkpoint in enumerate(checkpoints_path):
        
        print('\t Trial', checkpoint)
        # Trial folder name
        args.trainedweight_filepath = checkpoint
        net = Models(args, loader).model
        
        # Replace inplace of ReLU to False
        replace_layers(net)
        
        # For each element in loader
        save = output + checkpoint.split("/")[-2]
        os.makedirs(save, exist_ok=True)
        for step, data in enumerate(loader.TRAIN_DATASET):
            # Load original input
            in_tensor, label, clf_features = data["data"].to(device).unsqueeze(dim = 0).unsqueeze(dim = 0), data["labels"].squeeze().cpu().numpy(), data['clc_features'].to(device)
            in_numpy = in_tensor.squeeze().cpu().numpy()
            # Construct the saliency object. This alone doesn't do anything.
            integrated_gradients = saliency.IntegratedGradients()

            # Baselines : Min / Zero / Max values
            #baseline_min = np.full(in_numpy.shape, np.min(in_numpy))
            baseline_zero = np.zeros_like(in_numpy)
            #baseline_max = np.full(in_numpy.shape, np.max(in_numpy))
            
            # Arguments for call_model_function
            call_model_args = {'network': net, 'target':label, 'clf_features': clf_features}
            prediction = net(in_tensor, clf_features)[1].cpu().detach().numpy()
            # Compute Integrated Gradients for each baseline
            #attr_min = integrated_gradients.GetMask(x_value=in_numpy, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_min, x_steps=200, batch_size=1)
            attr_zero_ = integrated_gradients.GetMask(x_value=in_numpy, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_zero, x_steps=200, batch_size=1)
            #attr_max = integrated_gradients.GetMask(x_value=in_numpy, call_model_function=call_model_function, call_model_args=call_model_args, x_baseline=baseline_max, x_steps=200, batch_size=1)

            # Create a combined attribution map (average of all)
            #attr_mean = np.mean([attr_min, attr_zero, attr_max], axis=0)
            attr_zero_ = resize(attr_zero_, (237, 512, 512))
            ig_mask = np.zeros((attr_zero_.shape[2], attr_zero_.shape[1], attr_zero_.shape[0]))
            for i in range(attr_zero_.shape[0]):
                ig_mask[:,:,i] = attr_zero_[i,:,:]
            
            # Get the original file name and affine
            original_file = data['file_name']
            filename = original_file[original_file.rfind('\\')+1 : original_file.rfind('.')] + "_" + str(np.argmax(label)) + "_" + str(np.argmax(prediction))
            affine = data["affine"].squeeze()
            
            # Save all attribution maps (original & absolute)
            os.makedirs(save + '/Raw_attrs_IG/', exist_ok=True)
            save_file = save + '/Raw_attrs_IG/' + filename
            affine = data["affine"].squeeze()
    
            # Save attribution map as .nii file
            ni_img = nib.Nifti1Image(ig_mask, affine=affine)
            nib.save(ni_img, save_file)
            
            # Compute & Save absolute if needed
            if do_absolute:
                grads_abs = np.abs(ig_mask)
                save_abs = save + '/Raw_attrs(absolute)_IG/' + filename
                os.makedirs(save + '/Raw_attrs(absolute)_IG/', exist_ok=True)
                ni_img_abs = nib.Nifti1Image(grads_abs, affine=affine)
                nib.save(ni_img_abs, save_abs)
            
            
        # Free memory
        #del net, loader, label