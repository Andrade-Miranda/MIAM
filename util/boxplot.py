#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:49:41 2022

@author: gustavo
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pathCNN='/home/gustavo/Data/results/Hektor2021/predictions/FinalResults/CNN_h+VIT_n_Ensemble/metrics.csv'
# pathMCNN='/home/gustavo/Data/results/Hektor2021/predictions/FinalResults/MCNN_h+VIT_n_Ensemble/metrics.csv'
# pathUNETR='/home/gustavo/Data/results/Hektor2021/predictions/FinalResults/UNETR_Ensemble/metrics.csv'
# pathSwin='/home/gustavo/Data/results/Hektor2021/predictions/FinalResults/SwinTrans3D_Ensemble/metrics.csv'


# dataCNN = pd.read_csv(pathCNN)
# dataMCNN =pd.read_csv(pathMCNN)
# dataUNETR =pd.read_csv(pathUNETR)
# dataSwin =pd.read_csv(pathSwin)

# dataCNN = dataCNN.drop(range(0,25))
# dataCNN = dataCNN.drop([30,32,33,46])
# dataCNN = dataCNN.drop(range(69,101))

# dataMCNN = dataMCNN.drop(range(0,25))
# dataMCNN = dataMCNN.drop([30,32,33,46])
# dataMCNN = dataMCNN.drop(range(69,101))

# dataUNETR = dataUNETR.drop(range(0,25))
# dataUNETR = dataUNETR.drop([30,32,33,46])
# dataUNETR = dataUNETR.drop(range(69,101))

# dataSwin = dataSwin.drop(range(0,25))
# dataSwin = dataSwin.drop([30,32,33,46])
# dataSwin = dataSwin.drop(range(69,101))


# # Draw a vertical boxplot grouped 
# # by a categorical variable:
# #create your own color array
# my_colors = ["#4285f4", "#ea4335", 
#              "#34a853", "#ffff00"]
  
# # add color array to set_palette
# # function of seaborn
# sns.set_palette( my_colors ) 

# sns.boxplot(data=[dataCNN["dice"], dataMCNN["dice"], dataUNETR["dice"],dataSwin["dice"]], orient="v")


########ALL models#############
import os
from os import listdir
from os.path import isfile, join
predictionPath='/Users/gustavoandrade/Library/CloudStorage/GoogleDrive-gxandrade.miranda@gmail.com/Mi unidad/0_extra/results/Hecktor/predictions/FinalResults/Predictions-Final'
onlyfiles = [f for f in listdir(predictionPath) if isfile(join(predictionPath, f))]

data=[]
for i,files in enumerate(onlyfiles):
    data.append(pd.read_csv(os.path.join(predictionPath,files)))
    data[i]=data[i].assign(Model=files.split('.')[0])
    data[i].rename(columns={'dice': 'Dice', 'msd': 'ASSD'}, inplace=True)

Alldata=pd.concat(data,ignore_index=True)
my_colors = ["#4285f4", "#ea4335", 
             "#34a853", "#ffff00"]
  
# add color array to set_palette
# function of seaborn
sns.set(font_scale=1.8)
sns.set_style("whitegrid")
sns.set_palette( my_colors )
    

sns.boxplot(y=Alldata['Dice']*100,x=Alldata['Model'] ,orient="v",width=0.1)
sns.swarmplot(y=Alldata['Dice']*100,x=Alldata['Model'] ,orient="v",marker="x", linewidth=0.5,color='gray')

sns.boxplot(x=Alldata['Dice']*100,y=Alldata['Model'] ,orient="h")
sns.swarmplot(x=Alldata['Dice']*100,y=Alldata['Model'] ,orient="h",color='gray')

sns.boxplot(y=Alldata['ASSD'],x=Alldata['Model'] ,orient="v",width=0.2)
sns.swarmplot(y=Alldata['ASSD'],x=Alldata['Model'] ,orient="v",marker="x", linewidth=1,color='gray',)




##############NFOLDS HEKTOR########################################################################"""
listFolds=[30,31,32,33,34,35,36,37,38,39]
pathCNN='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/CNN_h+VIT_n/CNN_h+VIT_nF'
pathMCNN='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/MCNN_h+VIT_n/MCNN_h+VIT_nF'
pathUNETR='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/UNETR/UNETRF'
pathSwin='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/SwinTrans3D/SwinTrans3DF'

def plot_Nfolds_Boxplots(listFolds,pathCNN,pathMCNN,pathUNETR,pathSwin):
    for i in listFolds:
        pathCNN=pathCNN+str(i)+'/metrics.csv'
        pathMCNN=pathMCNN+str(i)+'/metrics.csv'
        pathUNETR=pathUNETR+str(i)+'/metrics.csv'
        pathSwin=pathSwin+str(i)+'/metrics.csv'


        dataCNN = pd.read_csv(pathCNN)
        dataMCNN =pd.read_csv(pathMCNN)
        dataUNETR =pd.read_csv(pathUNETR)
        dataSwin =pd.read_csv(pathSwin)


        dataCNN=dataCNN.assign(Model="CNN+VIT-{B}/1")
        dataMCNN=dataMCNN.assign(Model="MCNN+VIT-{B}/1")
        dataUNETR=dataUNETR.assign(Model="UNETR")
        dataSwin=dataSwin.assign(Model="Swin UNETR")
    
        Alldata=pd.concat([dataCNN, dataMCNN,dataUNETR,dataSwin], ignore_index=True)
        Alldata.rename(columns={'dice': 'Avg Dice', 'msd': 'ASSD'}, inplace=True)
        # Draw a vertical boxplot grouped 
        # by a categorical variable:
        #create your own color array
        my_colors = ["#4285f4", "#ea4335", 
             "#34a853", "#ffff00"]
  
        # add color array to set_palette
        # function of seaborn
        sns.set(font_scale=1.8)
        sns.set_style("whitegrid")
        sns.set_palette( my_colors )
    

        sns.boxplot(y=Alldata['Avg Dice']*100,x=Alldata['Model'] ,orient="v",width=0.5)
        sns.swarmplot(y=Alldata['Avg Dice']*100,x=Alldata['Model'] ,orient="v",marker="x", linewidth=1,color='gray')

        sns.boxplot(y=Alldata['ASSD'],x=Alldata['Model'] ,orient="v",width=0.5)
        sns.swarmplot(y=Alldata['ASSD'],x=Alldata['Model'] ,orient="v",marker="x", linewidth=1,color='gray',)






















