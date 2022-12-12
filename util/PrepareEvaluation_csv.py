#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:38:20 2022

@author: gustavoandrade
"""

import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
# Import Module
import os
from os import listdir
from os.path import isfile, join

predictionPath='/Users/gustavoandrade/Library/CloudStorage/GoogleDrive-gxandrade.miranda@gmail.com/Mi unidad/0_extra/results/Hecktor/predictions/FinalResults/Predictions-Final'


onlyfiles = [f for f in listdir(predictionPath) if isfile(join(predictionPath, f))]

data=[]
for i,files in enumerate(onlyfiles):
    data.append(pd.read_csv(os.path.join(predictionPath,files)))
    data[i].drop(['label','jaccard','precision','recall','fpr','fnr','vs','hd','msd','mdsd','stdsd','hd95'], axis=1, inplace=True)
    data[i]=data[i].assign(Task='T1')
    data[i]=data[i].assign(Algorithm=files.split('.')[0])
    data[i].rename(columns={'filename':'TestCase','dice': 'MetricValue'}, inplace=True)
    data[i] = data[i][['Task', 'TestCase', 'Algorithm', 'MetricValue']]
    data[i]['TestCase']=[fil.split('/')[-1] for fil in data[i]['TestCase']]

Alldata=pd.concat(data,ignore_index=True)
    
Alldata=Alldata.sort_values(by=['TestCase','Algorithm'])

#If include assD as metric    
# data1=[]
# for i,files in enumerate(onlyfiles):
#     data1.append(pd.read_csv(os.path.join(predictionPath,files)))
#     data1[i].drop(['label','jaccard','precision','recall','fpr','fnr','vs','hd','dice','mdsd','stdsd','hd95'], axis=1, inplace=True)
#     data1[i]=data1[i].assign(Task='T2')
#     data1[i]=data1[i].assign(Algorithm=files.split('.')[0])
#     data1[i].rename(columns={'filename':'TestCase','msd': 'MetricValue'}, inplace=True)
#     data1[i] = data1[i][['Task', 'TestCase', 'Algorithm', 'MetricValue']]
#     data1[i]['TestCase']=[fil.split('/')[-1] for fil in data1[i]['TestCase']]