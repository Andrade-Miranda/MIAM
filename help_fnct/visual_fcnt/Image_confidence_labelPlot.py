#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:52:09 2023

@author: gustavo
"""
import numpy as np
import skimage
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.exposure import rescale_intensity
import nibabel as nib
import os
import matplotlib.pyplot as plt
from scipy import ndimage
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
            im=np.fliplr(np.rot90(nifti_loadedImg.get_fdata()[:,:,slicAx],k=-1, axes=(0, 1)))
            image_numpyImg.append(im)

    return image_numpyImg,nifti_loadedImg
        

def drawContour(img,seg,color):

    # Create structuring element that defines the neighbourhood for morphology
    selem = skimage.morphology.disk(1)

    # Mask for edges of segment 1 and segment 2
    # We are basically looking for pixels with value 1 in the segmented image within a radius of 1 pixel of a black pixel...
    # ... then the same again but for pixels with a vaue of 2 in the segmented image within a radius of 1 pixel of a black pixel
    seg=skimage.morphology.remove_small_holes(seg,area_threshold=15)
    
    out=mark_boundaries(img, np.uint8(seg*1), color=color,mode='inner',outline_color=color)
    
    return out

def overlapping(image, heatmap):
    image=np.repeat(image[:,:,np.newaxis], 3, axis=-1)
    image = rescale_intensity(image, in_range=(np.min(image),np.max(image)), out_range=(0,1))
    # Normalize the heatmap values to be in the range [0, 1]
    #heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # Convert the heatmap to a color representation (using a colormap, e.g., 'jet')
    heatmap_color = plt.cm.jet(heatmap)
    # Blend the heatmap with the original image
    alpha = 0.25  # Adjust the transparency of the heatmap
    blended_image = (1 - alpha) * image + alpha * heatmap_color[:, :, :3]  # Use only RGB channels
    return blended_image



def boundaries(img,labels,colors):
    label_structure = np.ones((3, 3))
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    if img.shape[-1]!=3:
        img=np.repeat(img[:,:,np.newaxis], 3, axis=-1)
    for lab in unique_labels[1:]:
        color=colors[int(lab)-1]
        seg = drawContour(img,(labels==lab),color)
        img=seg
    
    return img# draw contour 1 in red
        

def boundariesv2(data,colors):
    img=data[0]
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    img=np.repeat(img[:,:,np.newaxis], 3, axis=-1)
    
    for labels,color in zip(data[1:],colors):
        seg = drawContour(img,labels,color)
        img= seg
    
    return img# draw contour 1 in red




models=dict()

pathlabel='/home/gustavo/Downloads/images/test/label'
pathimage='/home/gustavo/Downloads/images/test/images'
softmaxPath='/home/gustavo/Downloads/images/test/segmentation'#'/home/gustavo/Downloads/images/test/predictions'
outpath='/home/gustavo/Downloads/images/test/output2'
#NiftiHektor=['10103_1000103.nii.gz','20025_20025.nii.gz','20025_200251.nii.gz','20036_20036.nii.gz','20036_200361.nii.gz','20154_20154.nii.gz','20154_201541.nii.gz','20154_20154.nii.gz','20154_201541.nii.gz']
NiftiHektor=['10103_1000103.nii.gz','20001_20001.nii.gz','20001_200011.nii.gz','20036_20036.nii.gz','20036_200361.nii.gz']
#sliceAx=[9,12,7,7,7,7]

sliceAx=[12,7,7,7,7]

Models={'MCNN_h+VIT_n':(0.34, 0.86, 0.668),
        'Unet':(0.86, 0.371, 0.34),
        'unetr':(0.86, 0.34, 0.755),
        'Swin':(0.457, 0.86, 0.34),               
        'CNN_h+VIT_n':(0.52, 0.34, 0.86),
        'nnFormer':(0.34, 0.606, 0.86),
        'MedNeXt':(0.86, 0.817, 0.34)}


for file in range(len(NiftiHektor)):
    slice=int(sliceAx[file])
    filename=NiftiHektor[file].split('.')[0]
    
    for eachModel in Models.keys():
        image_numpyImg=[]
        colors=[]
        Image=os.path.join(pathimage, filename+'_0000.nii.gz')
        Seg=os.path.join(pathlabel, filename+'.nii.gz')
        softmax=os.path.join(softmaxPath,eachModel,filename+'.nii.gz')
        colors.append((0,0,1))
        colors.append((0,1,0))
        image_numpyImg,_=load_Nii([Image,Seg,softmax],axis=-1,slicAx=slice)

        #blending=overlapping(image_numpyImg[0], image_numpyImg[2])
        #out=boundaries(blending,image_numpyImg[1],colors)
        out=boundaries(image_numpyImg[0],image_numpyImg[1],colors)
        out=boundaries(out,np.uint8(image_numpyImg[-1]),[Models[eachModel]])

        
        imsave(os.path.join(outpath,filename+'-'+eachModel+'-'+str(slice)+'.png'),np.uint8(out*255))