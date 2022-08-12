#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:32:06 2021

@author: gustavo
"""

import os
import nibabel as nib
import ntpath
import argparse
import numpy as np
import re



def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--dataroot',type=str, default='/home/gustavo/Code/pytorch-CycleGAN-and-pix2pix/datasets/T1_T2Cropped/trainT1', help='path to .nii images 3D')
    parser.add_argument('--datasave', type=str, default='./datasets/T1/img', help='stores the results 2D') 
    parser.add_argument('--suffix', type=str, default='.nii.gz', help='suffix to replace') 
    parser.add_argument('--numFiles', type=int, default=0, help='0 represent all files') 
    
    return parser


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_Nii(imgpath_original,savepth,files,suffix):
    reference_nifti_loaded=nib.load(imgpath_original)
    image_numpy=reference_nifti_loaded.get_fdata()
    Omit_seg=files.replace(suffix,'')
    for i in range(image_numpy.shape[-1]):
        fname_out=(os.path.join(savepth,Omit_seg)+"_%03d.nii.gz") %i
        nib.save(nib.Nifti1Image(image_numpy[:,:,i], None, reference_nifti_loaded.header), fname_out)
    

# Function to extract all the numbers from the given string
def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return array



if __name__=='__main__':
    opt=initialize().parse_args()
    savepth=opt.datasave
    niidirpath=opt.dataroot
    suffix=opt.suffix
    numfiles=opt.numFiles
    mkdir(savepth)
    
    filenames=sorted([(int(getNumbers(fname)[0]),fname) for fname in os.listdir(niidirpath) if not fname.startswith('.')])
    
    if numfiles==0:
        numfiles=len(filenames)
    
    cont=0
    for _,files in filenames:
        if cont<numfiles:
            imgpath_original=os.path.join(niidirpath,files)
            save_Nii(imgpath_original,savepth,files,suffix)
        else:
            break
        cont+=1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
