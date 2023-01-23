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
import matplotlib.pyplot as plt
import skimage.filters.rank
import skimage.morphology
from skimage.io import imsave

def load_Nii(data,axis=0,slicAx=90):
    image_numpyImg=[]
    for img in data:
        nifti_loadedImg=nib.load(img)
        if axis==0:
            im=np.fliplr(np.rot90(nifti_loadedImg.get_fdata()[slicAx,:,:], k=1, axes=(0, 1)))
            image_numpyImg.append(im)
        elif axis==1:
            im=np.fliplr(np.rot90(nifti_loadedImg.get_fdata()[:,slicAx,:], k=1, axes=(0, 1)))
            image_numpyImg.append(im)
        else:
            im=np.fliplr(np.rot90(nifti_loadedImg.get_fdata()[:,:,slicAx], k=1, axes=(0, 1)))
            image_numpyImg.append(im)

    return image_numpyImg,nifti_loadedImg
        

def drawContour(img,seg,color):

    # Create structuring element that defines the neighbourhood for morphology
    #selem = skimage.morphology.disk(1)

    # Mask for edges of segment 1 and segment 2
    # We are basically looking for pixels with value 1 in the segmented image within a radius of 1 pixel of a black pixel...
    # ... then the same again but for pixels with a vaue of 2 in the segmented image within a radius of 1 pixel of a black pixel
    #seg= (skimage.filters.rank.minimum(np.uint8(seg),selem) == 0) & (skimage.filters.rank.maximum(np.uint8(seg), selem) == 1)
    
    out=mark_boundaries(img, np.uint8(seg*1), color=color)
    
    return out


def boundaries(data,colors):
    img=data[0]
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    img=np.repeat(img[:,:,np.newaxis], 3, axis=-1)
    
    for labels,color in zip(data[1:],colors):
        seg = drawContour(img,labels,color)
        img= seg
    
    return img# draw contour 1 in red
        












models=dict()

pathlabel='/home/gustavo/Code/Git_workspace/MIAM/nnUNet/data/nnUnet_raw/nnUNet_raw_data/Task003_HektorTest/labelsTr'
pathimage='/home/gustavo/Code/Git_workspace/MIAM/nnUNet/data/nnUnet_raw/nnUNet_raw_data/Task003_HektorTest/imagesTr'
hektorPath='/home/gustavo/Data/results/Hektor2021/predictions/FinalResults'
outpath='/home/gustavo/Code/Git_workspace/Ranking/Figures/Paper_Ilustrations/hektor'
NiftiHektor=['HektorTest2021-CHUP042_00014-59','HektorTest2021-CHUP048_00019-73',
             'HektorTest2021-CHUP052_00023-93','HektorTest2021-CHUV027_00074-85']

Models={'MCNN_h+VIT_n_Ensemble':(0.34,0.829,0.86),'MCNN_h+VIT_s_Ensemble':(0.34,0.606,0.86),'MCNN_h+VIT_cv_Ensemble':(0.34,0.383,0.86),
        'Unet_Ensemble':(0.86,0.371,0.34),'MCNN_h_Ensemble':(0.86,0.594,0.34),'nnUNet_Ensemble':(0.86,0.817,0.34),
        'UNETR_Ensemble':(0.52,0.34,0.86),'VIT_m_Ensemble':(0.743,0.34,0.86),'VIT_s_Ensemble':(0.86,0.34,0.755), 'SwinTrans3D_Ensemble':(0.86,0.34,0.532),               
        'CNN_h+VIT_n_Ensemble':(0.457,0.86,0.34),'Transfuse_Ensemble':(0.34,0.86,0.445),'Swinfuse_Ensemble':(0.34,0.86,0.668),
        'nnFormer_Ensemble':(0.68,0.86,0.34), 'GT':(1,0,1)}


TypeNetworks={'cnnBased':['Unet_Ensemble','MCNN_h_Ensemble','nnUNet_Ensemble'],
'FullTrans':['nnFormer_Ensemble'],
'Transform':['UNETR_Ensemble','VIT_m_Ensemble','VIT_s_Ensemble','SwinTrans3D_Ensemble'],   
'MultiVit':['MCNN_h+VIT_n_Ensemble','MCNN_h+VIT_cv_Ensemble','MCNN_h+VIT_s_Ensemble'],
'OneVit':['Transfuse_Ensemble','Swinfuse_Ensemble','CNN_h+VIT_n_Ensemble'],
'best':['MCNN_h+VIT_s_Ensemble','nnUNet_Ensemble']}

for file in NiftiHektor:
    sliceAx=int(file.split('-')[-1])
    filename="-".join([x for x in file.split('-')[:-1]])
    
    for TypeNet in TypeNetworks.keys():
        image_numpyImg=[]
        data=[]
        colors=[]
        Image=os.path.join(pathimage, filename+'_0000.nii.gz')
        Seg=os.path.join(pathlabel, filename+'.nii.gz')
        data.append(Image)
        for eachModel in TypeNetworks[TypeNet]:
            SegModel=os.path.join(hektorPath,eachModel,filename+'.nii.gz')
            data.append(SegModel)
            colors.append(Models[eachModel])
        data.append(Seg)   
        colors.append(Models['GT'])
        image_numpyImg,_=load_Nii(data,axis=0,slicAx=sliceAx)
        out=boundaries(image_numpyImg,colors)
        
        imsave(os.path.join(outpath,filename+'-'+TypeNet+'.png'),np.uint8(out*255))

        
        
    
       
    
    

