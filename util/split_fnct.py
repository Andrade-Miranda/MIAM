#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:19:17 2022

@author: gustavo
"""

#from sklearn.model_selection import KFold
from collections import OrderedDict
import numpy as np

def HEKTOR_Multicenter_split(all_keys_sorted):  # FOR HEKTOR
    a=[[],[],[],[],[]]
    cont=0
    for x in ['CHGJ','CHMR','CHUM','CHUP','CHUS']:
        for i in all_keys_sorted:
            if x in i:
                a[cont].append(str(i))
        cont+=1
    output=[OrderedDict({'train': np.array(a[1]+a[2]+a[3]+a[4]),'val': np.array(a[0])}),
            OrderedDict({'train': np.array(a[2]+a[3]+a[4]+a[0]),'val': np.array(a[1])}),
            OrderedDict({'train': np.array(a[3]+a[4]+a[0]+a[1]),'val': np.array(a[2])}),
            OrderedDict({'train': np.array(a[4]+a[0]+a[1]+a[2]),'val': np.array(a[3])}),
            OrderedDict({'train': np.array(a[0]+a[1]+a[2]+a[3]),'val': np.array(a[4])})
                        ]
    return output


def HEKTOR_splitprogressive(all_keys_sorted,num_fold):  # FOR HEKTOR
    a=[[],[],[],[],[]]
    cont=0
    for x in ['CHGJ','CHMR','CHUM','CHUP','CHUS']:
        for i in all_keys_sorted:
            if x in i:
                a[cont].append(str(i))
        cont+=1
    val=[]
    for i in range(len(a)):
        for j in range(4):# i will take 4 from each center
            val.append(a[i].pop())
    data=[]
    output=[]
    for n_folds in range(num_fold):
        for n_split in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170]:
            for i in range(len(a)):
                if n_split>70:
                    data.extend(np.random.choice(a[i],size=n_split//5,replace=True))
                else:
                    data.extend(np.random.choice(a[i],size=n_split//5,replace=False))
            output.append(OrderedDict({'train': np.array(data),'val':np.array(val)}))
            data=[]

    return output