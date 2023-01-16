#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:52:09 2023

@author: gustavo
"""

models=dict()


Models={'MCNN+ViTv':(0.34,0.829,0.86),'MCNN+ViT-{s}-B/1}$-c}':{0.34,0.606,0.86},'MCNN+ViT-{m}-B/1}$-c}{rgb}':(0.34,0.383,0.86),
        'U-Net':(0.86,0.371,0.34),'MU-Net':(0.86,0.594,0.34),'nn-UNet':(0.86,0.817,0.34)}
# \definecolor{nnFormer-c}{rgb}{0.68,0.86,0.34}
# \definecolor{$\mathregular{CNN+ViT-{v}-B/1}$-c}{rgb}{0.457,0.86,0.34}
# \definecolor{3D-Transfuse-c}{rgb}{0.34,0.86,0.445}
# \definecolor{3D-Swinfuse-c}{rgb}{0.34,0.86,0.668}

# \definecolor{$\mathregular{UNETR-{v}}$-c}{rgb}{0.52,0.34,0.86}
# \definecolor{$\mathregular{UNETR-{s}}$-c}{rgb}{0.743,0.34,0.86}
# \definecolor{$\mathregular{UNETR-{m}}$-c}{rgb}{0.86,0.34,0.755}
# \definecolor{Swin UNETR-c}{rgb}{0.86,0.34,0.532}

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





if __name__=='__main__':
    opt=initialize().parse_args()


    dataImg=opt.data
    dataGT=opt.GT
    dataUNETR=opt.UNETR
    dataCNN=opt.CNN
    dataMCNN=opt.MCNN
    dataSwin=opt.Swin

    colors=[(1,0,0),(0,0,1),(0,1,0),(0.9,0.6,0.9)]
    results=[dataImg,dataGT,dataMCNN,dataCNN,dataUNETR,dataSwin]    

    image_numpyImg,reference_nifti_loaded=load_Nii(results)
    out,masks=boundaries(image_numpyImg,colors)
    if type(masks)is list:
        j=1
        for msk in masks:
            nib.save(nib.Nifti1Image(msk*j, None, reference_nifti_loaded.header),os.path.join(opt.outputImg,'labelPred'+str(j)+'.nii.gz'))
            j+=1
    else:
        nib.save(nib.Nifti1Image(msk, None, reference_nifti_loaded.header),os.path.join(opt.outputImg,'labelPred.nii.gz'))
