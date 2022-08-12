#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 00:01:54 2022

@author: gustavoandrade
"""
import numpy as np
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.exposure import rescale_intensity
import nibabel as nib
import os
import argparse



def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data',type=str, default='/Users/gustavoandrade/Dropbox/predict/HektorTest2021-CHUP025_00000.nii.gz', help='path to .nii images')
    parser.add_argument('--GT',type=str, default='/Users/gustavoandrade/Dropbox/predict/HektorTest2021-CHUP025_00000_labelTrue.nii.gz', help='path to .nii groundtruth')
    parser.add_argument('--MCNN',type=str, default='/Users/gustavoandrade/Dropbox/predict/MCNN/10/HektorTest2021-CHUP025_00000_labelPred.nii.gz', help='path to .nii images')
    parser.add_argument('--CNN', type=str, default='/Users/gustavoandrade/Dropbox/predict/CNN/10/HektorTest2021-CHUP025_00000_labelPred.nii.gz', help='stores the nii concatenation') 
    parser.add_argument('--UNETR', type=str, default='/Users/gustavoandrade/Dropbox/predict/UNETR/10/HektorTest2021-CHUP025_00000_labelPred.nii.gz', help='stores the nii concatenation') 
    parser.add_argument('--Swin', type=str, default='/Users/gustavoandrade/Dropbox/predict/swin/10/HektorTest2021-CHUP025_00000_labelPred.nii.gz', help='stores the nii concatenation') 
    parser.add_argument('--outputImg', type=str, default='/Users/gustavoandrade/Dropbox/predict', help='stores the nii concatenation') 
    parser.add_argument('--colors',nargs='+',default=[(1,0,0),(0,0,1),(0,1,0),(0.9,0.6,0.9)],help='colors by default')
    
    return parser



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

    
        








