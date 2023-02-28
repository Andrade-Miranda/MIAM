#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:39:37 2021

@author: gustavo
"""


import sys
# setting path


import nibabel
import logging
import os
import re
import argparse

import json


class FakeDict(dict):
    def __init__(self, items):
        self['something'] = 'something'
        self._items = items
    def items(self):
        return self._items


def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--root_dir',type=str, default='/home/gustavo/nnUNet/data/nnUnet_raw/nnUNet_raw_data/Task500_Hektor', help='root directory')
    parser.add_argument('--name',type=str, default='Hektor2021', help='dataset name')
    parser.add_argument('--description', type=str, default='Hektor dataset CT and PET', help='dataset description') 
    parser.add_argument('--tensorImageSize', type=str, default='4D', help='Image size') 
    parser.add_argument('--modality', type=str, default='ct-pt', help='0 represent patient id') #flair-t1-t1ce-t2
    parser.add_argument('--labels', type=str, default='Tumor', help='0 represent patient id') #TC-WT-ET
    parser.add_argument('--option', type=int, default='0', help='0 only train, 1 train and val, 2 train and test') 
    
    return parser



if __name__=='__main__':
    
    opt = initialize().parse_args()
    last_folder='.'+'/'
    JSONFile={}
    JSONFile['name']=opt.name
    JSONFile['description']=opt.description
    JSONFile['tensorImageSize']=opt.tensorImageSize
    modal=opt.modality.split('-')
    labels=opt.labels.split('-')
    JSONFile['modality']={str(i):modal[i] for i in range(len(modal))}
    JSONFile['labels']={str(i+1):labels[i] for i in range(len(labels))}

    array_of_dictTr = []
    array_of_dictVal = []
    array_of_dictTs = []

    
    if opt.option==0:
        imagesTr=os.path.join(last_folder,'imagesTr')
        labelsTr=os.path.join(last_folder,'labelsTr')
        for fname in os.listdir(os.path.join(opt.root_dir,'imagesTr')):
            name=fname.split('_')
            fname=name[0]+'_'+name[1]+'.nii.gz'
            if not fname.startswith('.') and not fname.endswith('csv'):
                array_of_dictTr.append({"image":os.path.join(imagesTr,fname),"label":os.path.join(labelsTr,fname)})
        JSONFile["numTraining"]=len(array_of_dictTr) 
        JSONFile['training']=array_of_dictTr
        JSONFile['test']=[]
        
    elif opt.option==1:
        imagesTr=os.path.join(last_folder,'imagesTr')
        labelsTr=os.path.join(last_folder,'labelsTr')
        for fname in os.listdir(os.path.join(opt.root_dir,'imagesTr')):
            if not fname.startswith('.') and not fname.endswith('csv'):
                array_of_dictTr.append({"image":os.path.join(imagesTr,fname),"label":os.path.join(labelsTr,fname)})
        JSONFile["numTraining"]=len(array_of_dictTr) 
        JSONFile['training']=array_of_dictTr
        
        imagesVal=os.path.join(last_folder,'imagesVal')
        labelsVal=os.path.join(last_folder,'labelsVal')
        for fname in os.listdir(os.path.join(opt.root_dir,'imagesVal')):
            if not fname.startswith('.') and not fname.endswith('csv'):
                array_of_dictVal.append({"image":os.path.join(imagesVal,fname),"label":os.path.join(labelsVal,fname)})
        JSONFile["numValidation"]=len(array_of_dictVal) 
        JSONFile['validation']=array_of_dictVal
        
    elif opt.option==2:
        imagesTr=os.path.join(last_folder,'imagesTr')
        labelsTr=os.path.join(last_folder,'labelsTr')
        for fname in os.listdir(os.path.join(opt.root_dir,'imagesTr')):
            if not fname.startswith('.') and not fname.endswith('csv'):
                array_of_dictTr.append({"image":os.path.join(imagesTr,fname),"label":os.path.join(labelsTr,fname)})
        JSONFile["numTraining"]=len(array_of_dictTr) 
        JSONFile['training']=array_of_dictTr        
        
        
        imagesTs=os.path.join(last_folder,'imagesTs')
        labelsTs=os.path.join(last_folder,'labelsTs')
        for fname in os.listdir(os.path.join(opt.root_dir,'imagesTs')):
            if not fname.startswith('.') and not fname.endswith('csv'):
                array_of_dictTs.append({"image":os.path.join(imagesTs,fname),"label":os.path.join(labelsTs,fname)})
        JSONFile["numTest"]=len(array_of_dictTs)
        JSONFile['test']=array_of_dictTs    

    
    outfile=os.path.join(opt.root_dir,'dataset.json')
    with open(outfile, 'w') as outfile:
        json.dump(JSONFile,outfile)

    



