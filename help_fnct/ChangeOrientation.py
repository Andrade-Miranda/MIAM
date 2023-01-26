#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:01:39 2022

@author: gustavo
"""

import nibabel as nib
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# https://nipy.org/nibabel/coordinate_systems.html
#The inverse of the affine gives the mapping from scanner to voxel¶
#That means that the inverse of the affine matrix gives the transformation 
#from scanner RAS+ coordinates to voxel coordinates in the image data.
#Now imagine we have affine array A for someones_epi.nii.gz, and affine array 
#B for someones_anatomy.nii.gz. A gives the mapping from voxels in the image 
#data array of someones_epi.nii.gz to millimeters in scanner RAS+. B gives the 
#mapping from voxels in image data array of someones_anatomy.nii.gz to the same 
#scanner RAS+. Now let’s say we have a particular voxel coordinate (i,j,k) in 
#the data array of someones_epi.nii.gz, and we want to find the voxel in 
#someones_anatomy.nii.gz that is in the same spatial position. Call this matching 
#voxel coordinate (i′,j′,k′) . We first apply the transform from someones_epi.nii.gz 
#voxels to scanner RAS+ (A) and then apply the transform from scanner RAS+ to voxels
#in someones_anatomy.nii.gz (B−1):


# Below works for orientation but issues with header (check guy for correct version)

def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``
    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)
 
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
 
    return ornt_transf, ornt_init, ornt_fin
 
def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)
    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array
 
    return nib.orientations.apply_orientation(data_array, ornt_transf)

def ChangesLabelsToBrats(pred):
    predtemp = np.zeros(np.shape(pred),dtype=np.float64)
    predtemp[pred==1.0]=2.0
    predtemp[pred==2.0]=1.0
    predtemp[pred==3.0]=4.0


    return(predtemp)
    
 
# I test the code by the following simple demo, and it works.

target_path='/home/gustavo/Code/Git_workspace/MIAM/nnUNet/data/nnUnet_raw/nnUNet_raw_data/Task004_BraTS2021Test/labelsTr'
init_path='/home/gustavo/Data/results/BratS2021/predictions/FinalResults/nnUNet_Ensemble'

 
onlyfiles = [f for f in listdir(target_path) if isfile(join(target_path, f))]

for file in onlyfiles:
    init_nii = nib.load(os.path.join(init_path,file))
    lits_data = ChangesLabelsToBrats(init_nii.get_fdata())  # shape = (512, 512, 123)

    kits_axcodes = tuple(nib.aff2axcodes(init_nii.affine))  # ('R', 'A', 'S')
    tatget_nii = nib.load(os.path.join(target_path,file))
    lits_axcodes = tuple(nib.aff2axcodes(tatget_nii.affine))  # ('I', 'P', 'L')

    new_lits_img = do_reorientation(lits_data, lits_axcodes, kits_axcodes)  # shape = (123, 512, 512) 

    header_info = init_nii.header
    new_image = nib.Nifti1Image(new_lits_img.astype(np.uint8), tatget_nii.affine, header_info)
    nib.save(new_image, os.path.join("/home/gustavo/Data/results/BratS2021/predictions/FinalResults/nnUNet",file))





