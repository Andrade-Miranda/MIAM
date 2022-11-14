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







##############

for i in [10]:
    pathCNN='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/CNN_h+VIT_n/CNN_h+VIT_nF'+str(i)+'/metrics.csv'
    pathMCNN='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/MCNN_h+VIT_n/MCNN_h+VIT_nF'+str(i)+'/metrics.csv'
    pathUNETR='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/UNETR/UNETRF'+str(i)+'/metrics.csv'
    pathSwin='/home/gustavo/Data/results/Hektor2021/predictions/nfolds1/SwinTrans3D/SwinTrans3DF'+str(i)+'/metrics.csv'


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






















