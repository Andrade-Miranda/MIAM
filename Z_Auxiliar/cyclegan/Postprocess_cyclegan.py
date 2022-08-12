#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:11:40 2021

@author: gustavo
"""

import os
import nibabel as nib
import argparse
import numpy as np
import re




def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--dataOriginal',type=str, default='../datasets/valT2_complet/', help='path to .nii images')
    parser.add_argument('--dataFakeImg',type=str, default='../Output/uNet19Cochlea/valT2/FakeImg', help='path to .nii images')
    parser.add_argument('--datasaveImg', type=str, default='../Output/uNet19Cochlea/valT2/FakeNiiImg', help='stores the nii concatenation') 

    
    return parser


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_Nii(dataImg,dataOrig,opt):
    image_numpyImg=[]
    for key in dataOrig:
        value=dataOrig[key]
        reference_nifti_loaded=nib.load(os.path.join(opt.dataOriginal,value))
        for valImg in dataImg[key]:
            nifti_loadedImg=nib.load(os.path.join(opt.dataFakeImg,valImg))
            image_numpyImg.append(nifti_loadedImg.get_fdata())

        
        outputImg=os.path.join(opt.datasaveImg,valImg.split('_CER')[0].lstrip().split(' ')[0]+valImg.split('.')[1].lstrip().split(' ')[0]+'.nii.gz')
        mkdir(opt.datasaveImg)
        
        nib.save(nib.Nifti1Image(np.moveaxis(np.asarray(image_numpyImg), [0,1], [-1,0]), None, reference_nifti_loaded.header),outputImg)
        image_numpyImg=[]
    

# Function to extract all the numbers from the given string
def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return array



if __name__=='__main__':
    opt=initialize().parse_args()


    filenamesImg=sorted([(int(getNumbers(fname)[0]),fname) for fname in os.listdir(opt.dataFakeImg) if not fname.startswith('.')])
    filenamesOrig=sorted([(int(getNumbers(fname)[0]),fname) for fname in os.listdir(opt.dataOriginal) if not fname.startswith('.')])

    
    dataImg = {filenamesImg[i][0]: [] for i in range(len(filenamesImg))}
    dataOrig = {filenamesOrig[i][0]: filenamesOrig[i][1] for i in range(len(filenamesOrig))}
    
    cont=0
    for filesImg in filenamesImg:
        dataImg[filesImg[0]].append(filesImg[1])
    save_Nii(dataImg,dataOrig,opt)
        

        
        
        
        
        
        
        
        
        
        
        
        
        