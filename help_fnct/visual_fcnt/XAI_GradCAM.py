# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:21:46 2022

@author: Wistan
"""


import os
import torch
import numpy as np
import nibabel as nib
import saliency.core as saliency
from saliency.core.base import CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS
from skimage.transform import resize

#from utils.XAI_utils import XAI_dataset
from models.Models import *
from utils.Tools import replace_layers
from skimage.transform import resize



#------------------------------------     PARAMETERS     ------------------------------------#


# Base paths
loading = './'


# List of folds
trials = {
            0 : 'layer4.0.conv1',
            1 : 'layer4.0.conv1',
            2 : 'layer4.0.conv1',
          }


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


# Expected keys for GradCAM
expected_keys = [CONVOLUTION_LAYER_VALUES, CONVOLUTION_OUTPUT_GRADIENTS]


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#------------------------------------     Hooks Functions     ------------------------------------#


conv_layer_outputs = {} 
def conv_layer_forward(m, i, o):
    # Move the channels number dim to the end
    conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, -1).detach().cpu().numpy()
def conv_layer_backward(m, i, o):
    # Move the channels number dim to the end
    conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, -1).detach().cpu().numpy()


#------------------------------------     Custom GradCAM     ------------------------------------#


# Function was modified to accept 3D input
def compute_GradCAM(x_value_1, x_value_2,
            call_model_function,
            call_model_args=None,
            should_resize=True,
            three_dims=True):
    
    x_value_1_batched = np.expand_dims(x_value_1, axis=0)
    x_value_2_batched = np.expand_dims(x_value_2, axis=0)
    data = call_model_function( x_value_1_batched, x_value_2_batched,
                                call_model_args=call_model_args,
                                expected_keys=expected_keys)
    
    weights = np.mean(data[CONVOLUTION_OUTPUT_GRADIENTS][0], axis=(0, 1, 2))
    grad_cam = np.zeros(data[CONVOLUTION_LAYER_VALUES][0].shape[0:3],
                        dtype=np.float32)

    # weighted average
    for i, w in enumerate(weights):
      grad_cam += w * data[CONVOLUTION_LAYER_VALUES][0][:, :, :, i]

    # pass through relu
    grad_cam = np.maximum(grad_cam, 0)

    # resize heatmap to be the same size as the input
    if should_resize:
      if np.max(grad_cam) > 0:
        grad_cam = grad_cam / np.max(grad_cam)
      grad_cam = resize(grad_cam, x_value_1.shape[:3])

    return grad_cam


#------------------------------------     call_model_function     ------------------------------------#


def call_model_function(images, clf_features,
                        call_model_args=None,
                        expected_keys=None):
    
    handle_forward = call_model_args['layer'].register_forward_hook(conv_layer_forward)
    handle_backward = call_model_args['layer'].register_full_backward_hook(conv_layer_backward)

    images = torch.tensor(images, device=device, dtype=torch.float32).unsqueeze(1)
    clf_features = torch.tensor(clf_features, device=device, dtype=torch.float32).unsqueeze(1)
    print(images.shape)
    print(clf_features.shape)
    images.requires_grad = True
    target_class_idx = call_model_args['target']
    output = call_model_args['network'](images, clf_features)
    target = output[0][0, output[0].argmax()]
    call_model_args['network'].zero_grad()
    target.backward(retain_graph=True)
    
    handle_forward.remove()
    handle_backward.remove()
    
    return conv_layer_outputs


#------------------------------------     GRADCAM     ------------------------------------#



def GradCAM(loader, checkpoints_path , output, do_absolute, args): 
    
    # For each trial (=fold)
    for c, checkpoint in enumerate(checkpoints_path):
        
        print('\t Trial', checkpoint)
        
        # Trial folder name
        args.trainedweight_filepath = checkpoint
        net = Models(args, loader).model
        
        # Replace inplace of ReLU to False
        replace_layers(net)
        
        # Save folder
        save = output + checkpoint.split("/")[-2]
        os.makedirs(save, exist_ok=True)
        
        # Look at all layers
        for i, d in enumerate(net.named_modules()):
            
            # If selected layer
            if (d[0] == trials[c]):
                print('\t\t', d[0])
                
                # Save layer and break for loop
                conv_layer = d[1]
                break
       
        # For each element in loader
        for step, data in enumerate(loader.TRAIN_DATASET):
            # Load original input
            in_tensor, label, clf_features = data["data"].to(device), data["labels"].squeeze().cpu().numpy(), data['clc_features'].to(device)
            in_numpy_1 = in_tensor.squeeze().cpu().numpy()
            in_numpy_2 = clf_features.squeeze().cpu().numpy()
            call_model_args = {'network': net, 'layer': conv_layer, 'target':label}
            
            prediction = net(in_tensor.unsqueeze(dim = 0).unsqueeze(dim = 0), clf_features.unsqueeze(dim = 0))[1].cpu().detach().numpy()
            # Compute the Grad-CAM mask
            grad_cam_mask_ = compute_GradCAM(x_value_1=in_numpy_1, x_value_2=in_numpy_2, call_model_function=call_model_function, call_model_args=call_model_args, should_resize=True, three_dims=False)
            # Resizing back to the original image dimension
            grad_cam_mask_ = resize(grad_cam_mask_, (237, 512, 512))
            grad_cam_mask = np.zeros((grad_cam_mask_.shape[2], grad_cam_mask_.shape[1], grad_cam_mask_.shape[0]))
            for i in range(grad_cam_mask_.shape[0]):
              grad_cam_mask[:,:,i] = grad_cam_mask_[i,:,:]
            # Get the original file name and affine
            original_file = data['file_name']
            filename = original_file[original_file.rfind('\\')+1 : original_file.rfind('.')] + "_" + str(np.argmax(label)) + "_" + str(np.argmax(prediction))
            os.makedirs(save + '/Raw_attrs_GradCAM/', exist_ok=True)
            save_file = save + '/Raw_attrs_GradCAM/' + filename
            affine = data["affine"].squeeze()
            # Save attribution map as .nii file
            ni_img = nib.Nifti1Image(grad_cam_mask, affine=affine)
            nib.save(ni_img, save_file)
            