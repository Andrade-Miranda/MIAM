#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:59:36 2021

@author: gustavo
"""

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
    parser.add_argument('--dataOriginal',type=str, default='../datasets/Complete_Data/valT2_complet/', help='path to .nii images')
    parser.add_argument('--dataFakeImg',type=str, default='../Output/results_e15/valT2/FakeImg', help='path to .nii images')
    parser.add_argument('--dataFakeSeg',type=str, default='../Output/results_e15/valT2/FakeSeg', help='path to .nii images')
    parser.add_argument('--datasaveImg', type=str, default='../Output/results_e15/valT2/FakeNiiImg', help='stores the nii concatenation') 
    parser.add_argument('--datasaveSeg', type=str, default='../Output/results_e15/valT2/FakeNiiSeg', help='stores the nii concatenation') 

    
    return parser


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_Nii(dataImg,dataSeg,dataOrig,opt):
    image_numpyImg=[]
    image_numpySeg=[]
    for key in dataOrig:
        value=dataOrig[key]
        reference_nifti_loaded=nib.load(os.path.join(opt.dataOriginal,value))
        for valImg,valSeg in zip(dataImg[key],dataSeg[key]):
            nifti_loadedImg=nib.load(os.path.join(opt.dataFakeImg,valImg))
            image_numpyImg.append(nifti_loadedImg.get_fdata())
            nifti_loadedSeg=nib.load(os.path.join(opt.dataFakeSeg,valSeg))
            image_numpySeg.append(nifti_loadedSeg.get_fdata())
        
        outputImg=os.path.join(opt.datasaveImg,valImg.split('_CER')[0].lstrip().split(' ')[0]+valImg.split('.')[1].lstrip().split(' ')[0]+'.nii.gz')
        outputSeg=os.path.join(opt.datasaveSeg,valSeg.split('_CER')[0].lstrip().split(' ')[0]+valSeg.split('.')[1].lstrip().split(' ')[0]+'.nii.gz')
        mkdir(opt.datasaveImg)
        mkdir(opt.datasaveSeg)
        
        nib.save(nib.Nifti1Image(np.moveaxis(np.asarray(image_numpyImg), [0,1], [-1,0]), None, reference_nifti_loaded.header),outputImg)
        nib.save(nib.Nifti1Image(np.moveaxis(np.asarray(image_numpySeg), [0,1], [-1,0]), None, reference_nifti_loaded.header),outputSeg)
        image_numpyImg=[]
        image_numpySeg=[]
    

# Function to extract all the numbers from the given string
def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return array



if __name__=='__main__':
    opt=initialize().parse_args()


    filenamesImg=sorted([(int(getNumbers(fname)[0]),fname) for fname in os.listdir(opt.dataFakeImg) if not fname.startswith('.')])
    filenamesSeg=sorted([(int(getNumbers(fname)[0]),fname) for fname in os.listdir(opt.dataFakeSeg) if not fname.startswith('.')])
    filenamesOrig=sorted([(int(getNumbers(fname)[0]),fname) for fname in os.listdir(opt.dataOriginal) if not fname.startswith('.')])

    
    dataImg = {filenamesImg[i][0]: [] for i in range(len(filenamesImg))}
    dataSeg = {filenamesSeg[i][0]: [] for i in range(len(filenamesSeg))}
    dataOrig = {filenamesOrig[i][0]: filenamesOrig[i][1] for i in range(len(filenamesOrig))}
    
    cont=0
    for filesImg,filesSeg in zip(filenamesImg,filenamesSeg):
        dataImg[filesImg[0]].append(filesImg[1])
        dataSeg[filesSeg[0]].append(filesSeg[1])
    save_Nii(dataImg,dataSeg,dataOrig,opt)
        

        
        
        
        
        
        
        
        
        
        
        
        
        