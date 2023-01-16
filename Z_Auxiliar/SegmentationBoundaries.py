#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:52:09 2023

@author: gustavo
"""
import numpy as np
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.exposure import rescale_intensity
import nibabel as nib
import os
import argparse


def load_Nii(data):
    image_numpyImg=[]
    for img in data:
        nifti_loadedImg=nib.load(img)
        image_numpyImg.append(nifti_loadedImg.get_fdata())

    return image_numpyImg,nifti_loadedImg
        

        
    

def boundaries(data,colors):
    img=data.pop(0)[:,:,:,0]
    groundtruth=data.pop(0)[:,:,:,0]
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    out=np.zeros((img.shape[0],img.shape[1],img.shape[2],3))
    mask=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    totalMsk=[]
    for i in range(img.shape[-2]):
        out[:,:,i,:] = mark_boundaries(img[:,:,i], groundtruth.astype('uint8')[:,:,i], color=(0.4, 0, 1))
        mask[:,:,i] = find_boundaries(groundtruth.astype('uint8')[:,:,i])
    i=0
    totalMsk.append(mask)
    for pred in data:
        mask=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
        for j in range(img.shape[-2]):
            out[:,:,j,:] = mark_boundaries(out[:,:,j,:], pred[:,:,j,0].astype('uint8'), color=colors[i])
            mask[:,:,j] = find_boundaries(pred[:,:,j,0].astype('uint8'))
        i+=1 
        totalMsk.append(mask)
    
    return out,totalMsk  



models=dict()

pathlabel='/Users/gustavoandrade/Downloads/labelsTr'
pathimage='/Users/gustavoandrade/Downloads/imagesTr'
NiftiHektor=['HektorTest2021-CHUP042_00014-100','HektorTest2021-CHUP048_00019-71',
             'HektorTest2021-CHUP052_00023-75','HektorTest2021-CHUV027_00074-69']

Models={'MCNN_h+VIT_n_Ensemble':(0.34,0.829,0.86),'MCNN_h+VIT_s_Ensemble':{0.34,0.606,0.86},'MCNN_h+VIT_cv_Ensemble':(0.34,0.383,0.86),
        'Unet_Ensemble':(0.86,0.371,0.34),'MCNN_h_Ensemble':(0.86,0.594,0.34),'nnUNet_Ensemble':(0.86,0.817,0.34),
        'UNETR_Ensemble':(0.52,0.34,0.86),'VIT_m_Ensemble':(0.743,0.34,0.86),'VIT_s_Ensemble':(0.86,0.34,0.755), 'SwinTrans3D_Ensemble':(0.86,0.34,0.532),               
        'CNN_h+VIT_n_Ensemble':(0.457,0.86,0.34),'Transfuse_Ensemble':(0.34,0.86,0.445),'Swinfuse_Ensemble':(0.34,0.86,0.668),
        'nnFormer_Ensemble':(0.68,0.86,0.34), 'GT':(0.5,0.5,0.5)}


cnnBased=['GT','Unet_Ensemble','MCNN_h_Ensemble','nnUNet_Ensemble']
FullTrans=['GT','nnFormer_Ensemble']
Transform=['GT','UNETR_Ensemble','VIT_m_Ensemble','VIT_s_Ensemble','SwinTrans3D_Ensemble']   
MultiVit=['GT','MCNN_h+VIT_n_Ensemble','MCNN_h+VIT_s_Ensemble','MCNN_h+VIT_cv_Ensemble']
OneVit=['GT','CNN_h+VIT_n_Ensemble','Transfuse_Ensemble','Swinfuse_Ensemble']
best=['GT','MCNN_h+VIT_s_Ensemble','nnUNet_Ensemble']

for file in NiftiHektor:
    sliceAx=file.split('_')[-1]
    image_numpy,reference_nifti_loaded=load_Nii(os.path.join(pathlabel,file))
    
    
out,masks=boundaries(image_numpyImg,colors)
if type(masks)is list:
    j=1
    for msk in masks:
        nib.save(nib.Nifti1Image(msk*j, None, reference_nifti_loaded.header),os.path.join(opt.outputImg,'labelPred'+str(j)+'.nii.gz'))
        j+=1
else:
       nib.save(nib.Nifti1Image(msk, None, reference_nifti_loaded.header),os.path.join(opt.outputImg,'labelPred.nii.gz'))
