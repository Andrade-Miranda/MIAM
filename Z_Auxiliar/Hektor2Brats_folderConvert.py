#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:41:32 2022

@author: gustavo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:14:12 2021

@author: gustavo
"""
import nibabel
import argparse

import sys
# setting path
sys.path.append('../util')
import preProcessing as PP

import os
import re
import shutil



def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--dataroot',type=str, default='/home/gustavo/Code/CNNTrans/datasets/resampled', help='path to .nii images 3D')
    parser.add_argument('--datasave', type=str, default='/home/gustavo/Code/CNNTrans/datasets/', help='stores the modalities concatenation') 
    parser.add_argument('--modal', type=str, default='ct-pt-gtvt', help='the modalities to read, seg has to be the last one') 
    parser.add_argument('--FoldName', type=str, default='Hektor2021', help='specify the data to convert') 
    parser.add_argument('--extension', type=str, default='.nii.gz', help='file extension')
    parser.add_argument('--selectedIds', dest='selectedIds',action='store_true',default=False, help='Take only parts of the patient depending of rate')
    

    #T1DUALin-src T1DUALout-src
    
    return parser


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

 # Function to extract all the numbers from the given string
def getNumbers(self,str):
    array = re.findall(r'[0-9]+', str)
    return array



if __name__=='__main__':
    
    ## initialization class pre-processing
    pProcess=initialize().parse_args()
    modalities=pProcess.modal.split('-')
    folderSave=os.path.join(pProcess.datasave,pProcess.FoldName)
    mkdir(folderSave)
    
    # load modalities in separate list
    data = [[] for i in range(len(modalities))]
    for fname in os.listdir(pProcess.dataroot): 
        if not fname.startswith('.') and not fname.endswith('csv'):
            file=fname.split('_')[0]
            cont=0
            for modal in modalities:
                if modal in fname.split('_')[1]:
                    break
                cont+=1  
            data[cont].append(fname)
    for i in range(len(modalities)):
        data[i].sort()
            

    for i in range(len(data[0])):
        filename=pProcess.FoldName+'-'+data[0][i].split('_')[0]
        folderOutput=os.path.join(folderSave,filename+'_%0*d'%(5,i))
        mkdir(folderOutput)
        for mod in range(len(modalities)):
            os.rename(os.path.join(pProcess.dataroot,data[mod][i]), os.path.join(folderOutput,filename+'_%0*d'%(5,i)+'_'+data[mod][i].split('_')[1]))
            #os.replace(os.path.join(pProcess.dataroot,data[mod][i]), os.path.join(folderOutput,filename+'_%0*d'%(5,i)+'_'+data[mod][i].split('_')[1]))
            #shutil.move(os.path.join(pProcess.dataroot,data[mod][i]), os.path.join(folderOutput,filename+'_%0*d'%(5,i)+'_'+data[mod][i].split('_')[1]))
     
    
    
    
    