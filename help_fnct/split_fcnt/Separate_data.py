#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:14:12 2021

@author: gustavo
"""
import nibabel
import argparse

import sys
import os
# setting path
sys.path.append('/home/gustavo/Code/Git_workspace/MIAM/help_fnct/prepro_fcnt')# to check
import preProcessing as PP



def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--dataroot',type=str, default='/home/gustavo/Data/dataset/ProstateDATA/Multireader_NIFTI_crop', help='path to .nii images 3D')
    parser.add_argument('--datasave', type=str, default='/home/gustavo/Code/Git_workspace/MIAM/nnUNet/data/nnUnet_raw/nnUNet_raw_data/Task2203_Prostate_multireader', help='stores the modalities concatenation') 
    parser.add_argument('--modal', type=str, default='T2-Prostate_ES', help='the modalities to read, seg has to be the last one') 
    parser.add_argument('--option', type=str, default='Train', help='specify the data to convert') 
    parser.add_argument('--extension', type=str, default='.nii.gz', help='file extension')
    parser.add_argument('--selectedIds', dest='selectedIds',action='store_true',default=False, help='Take only parts of the patient depending of rate')
    parser.add_argument('--rate', nargs='+', default= [1,0,0], help='take from folder only the % Id_ ')
    

    #T1DUALin-src T1DUALout-src
    
    return parser


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



if __name__=='__main__':
    
    ## initialization class pre-processing
    pProcess=PP.preProcessing(initialize().parse_args())
    
    
    ## use when we want to separate modalities in separate folders
    ### create subfolders to save pre-processed images
    if pProcess.option=='Train':
        images=os.path.join(pProcess.datasave,'imagesTr')
        labels=os.path.join(pProcess.datasave,'labelsTr')
        mkdir(images)
        mkdir(labels)
    elif pProcess.option=='val':
        images=os.path.join(pProcess.datasave,'imagesVal')
        labels=os.path.join(pProcess.datasave,'labelsVal')
        mkdir(images)
        mkdir(labels)
    elif pProcess.option=='test':
        images=os.path.join(pProcess.datasave,'imagesTs')
        labels=os.path.join(pProcess.datasave,'labelsTs')
        mkdir(images)
        mkdir(labels)

        
    # read each patient id, normalize data and crop if it is necessary
    idsPatients=pProcess.data
    idsPatients.sort()
    if pProcess.selectedIds:
        training = idsPatients[:int(len(idsPatients)*pProcess.rate[0])] #[1, 2, 3, 4, 5, 6, 7, 8]
        validation = idsPatients[int(len(idsPatients)*pProcess.rate[0]):int(len(idsPatients)*pProcess.rate[0])+int(len(idsPatients)*pProcess.rate[1])] #[9]
        testing = idsPatients[int(len(idsPatients)*pProcess.rate[0])+int(len(idsPatients)*pProcess.rate[1]):] #[10]
        if pProcess.option=='Train':
            idsPatients=training
        elif pProcess.option=='val': 
            idsPatients=validation
        elif pProcess.option=='test':
            idsPatients=testing
        
    position=0
    for id_ in idsPatients:
        # initialize list of data
        pProcess.exam_uploadVProstate(id_,position)
        #modal_normal,segmentation=pProcess.extractModal(id_)
        # save data in the correct folders
        #nibabel.save(modal_normal,os.path.join(images,id_[1]+'_'+id_[0]+pProcess.extension))
        #nibabel.save(segmentation,os.path.join(labels,id_[1]+'_'+id_[0]+pProcess.extension))
        position+=1

    
    
    
    
    
    
    
    
    
    
    
    
    
    
