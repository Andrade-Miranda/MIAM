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


def HEKTOR_splitprogressive2(all_keys_sorted,num_fold):  # FOR HEKTOR
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
        for n_split in range(0,10):
            for i in range(len(a)):
                if n_split>7:
                    data.extend(np.random.choice(a[i],size=10//5,replace=True))
                else:
                    data.extend(np.random.choice(a[i],size=10//5,replace=False))
            output.append(OrderedDict({'train': np.array(data),'val':np.array(val)}))
        data=[]

    return output

def Brats_splitprogressive2(all_keys_sorted,num_fold):
    a=[]
    badsamples=['BraTS2021_00008','BraTS2021_00009','BraTS2021_00011','BraTS2021_00012','BraTS2021_00014','BraTS2021_00017','BraTS2021_00019','BraTS2021_00021','BraTS2021_00024',
                'BraTS2021_00015','BraTS2021_00028','BraTS2021_00033','BraTS2021_00053','BraTS2021_00062','BraTS2021_00084','BraTS2021_00085','BraTS2021_00098',
                'BraTS2021_00102','BraTS2021_00104','BraTS2021_00108','BraTS2021_00110','BraTS2021_00111','BraTS2021_00113','BraTS2021_00116','BraTS2021_00140','BraTS2021_00155',
                'BraTS2021_00160','BraTS2021_00162','BraTS2021_00167','BraTS2021_00170','BraTS2021_00171','BraTS2021_00178','BraTS2021_00185','BraTS2021_00204',
                'BraTS2021_00210','BraTS2021_00212','BraTS2021_00218','BraTS2021_00219','BraTS2021_00222','BraTS2021_00228','BraTS2021_00142','BraTS2021_00143',
                'BraTS2021_00148','BraTS2021_00157','BraTS2021_00186','BraTS2021_00199','BraTS2021_00206','BraTS2021_00217','BraTS2021_00236','BraTS2021_00234',
                ]
    valdata=['BraTS2021_00016','BraTS2021_00026','BraTS2021_00031','BraTS2021_00044','BraTS2021_00054','BraTS2021_00056','BraTS2021_00061','BraTS2021_00063',
             'BraTS2021_00081','BraTS2021_00087','BraTS2021_00091','BraTS2021_000107','BraTS2021_00112','BraTS2021_00121','BraTS2021_00136','BraTS2021_00149',
             'BraTS2021_00192','BraTS2021_00195','BraTS2021_00196','BraTS2021_00227']
    for i in all_keys_sorted:
        if i in badsamples:
            pass
        elif i in valdata:
            pass
        elif i =='BraTS2021_00237':
            break
        else:
            a.append(i)
    data=[]
    output=[]
    for n_folds in range(num_fold):
        for n_split in range(0,10):
            data.extend(np.random.choice(a,size=10,replace=False))
            output.append(OrderedDict({'train': np.array(data),'val':np.array(valdata)}))
        data=[]

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
                    data.extend(np.random.choice(a[i],size=10//5,replace=True))
                else:
                    data.extend(np.random.choice(a[i],size=10//5,replace=False))
            output.append(OrderedDict({'train': np.array(data),'val':np.array(val)}))
            data=[]

    return output