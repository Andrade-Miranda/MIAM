#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:12:38 2021

@author: gustavo
"""

import os
import argparse
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt



def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--dataroot',type=str, default='/home/gustavo/Code/Synseg-Net/checkpoints/yh_cyclegan_imgandseg/loss_log.txt' )
    parser.add_argument('--datasave', type=str, default='/home/gustavo/Code/Synseg-Net/Output/uNet19Cochlea/plots', help='stores the plots') 
    
    return parser


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to extract all the numbers from the given string
def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return array


# errors: dictionary of error labels and values
def plot_current_errorsVSiterations(errors,datasave):
    epochs = errors.loc[:,'epoch']
    epochs = epochs.values
    D_A = errors.loc[:,'D_A']
    D_A = D_A.values
    G_A = errors.loc[:,'G_A']
    G_A = G_A.values
    cycle_A = errors.loc[:,'cycle_A']
    cycle_A = cycle_A.values
    
    x=np.arange(0,len(D_A))*100
    
    figA, axA = plt.subplots(3,1,constrained_layout=True)
    figA.suptitle("T1-T2 Loss vs iteration", fontsize=14)
    axA[0].plot(x,D_A,color='green')
    axA[0].set_xlabel('iterations')
    axA[0].set_ylabel('Loss')
    axA[0].set_title('T1 Discriminator - MSE Loss')

    axA[1].plot(x,cycle_A,color='red')
    axA[1].set_xlabel('iterations')
    axA[1].set_ylabel('Loss')
    axA[1].set_title('T1 Cycle - L1 Loss')

    
    axA[2].plot(x,G_A,color='blue')
    axA[2].set_xlabel('iterations')
    axA[2].set_ylabel('Loss')
    axA[2].set_title('T1 Generator - MSE Loss')
    plt.savefig(os.path.join(datasave,'LossVsiteration_T1-T2.pdf'))

    
    D_B = errors.loc[:,'D_B']
    D_B = D_B.values
    G_B = errors.loc[:,'G_B']
    G_B = G_B.values
    cycle_B = errors.loc[:,'cycle_B']
    cycle_B = cycle_B.values
    
    figB, axB = plt.subplots(3,1,constrained_layout=True)
    figB.suptitle("T2-T1 Loss vs iteration", fontsize=14)
    axB[0].plot(x,D_B,color='green')
    axB[0].set_xlabel('iterations')
    axB[0].set_ylabel('Loss')
    axB[0].set_title('T2 Discriminator - MSE Loss')


    axB[1].plot(x,cycle_B,color='red')
    axB[1].set_xlabel('iterations')
    axB[1].set_ylabel('Loss')
    axB[1].set_title('T2 Cycle - L1 Loss')


    
    axB[2].plot(epochs,G_B,color='blue')
    axB[2].set_xlabel('iterations')
    axB[2].set_ylabel('Loss')
    axB[2].set_title('T2 Generator - MSE Loss')
    plt.savefig(os.path.join(datasave,'LossVsiteration_T2-T1.pdf'))


    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle("Segmentation Loss vs iteration", fontsize=14)
    ax.set_title('Segmentation - Tversky Loss')
    ax.set_ylim(0, 3)
    Seg = errors.loc[:,'Seg']
    ax.plot(x,Seg,color='blue')
    plt.savefig(os.path.join(datasave,'LossVSepoch_Loss.pdf'))


# errors: dictionary of error labels and values
def plot_current_errorsVSepochs(errors,datasave):
    epochs = errors.loc[:,'epoch']
    epochs = epochs.values
    D_A = errors.loc[:,'D_A']
    D_A = D_A.values
    G_A = errors.loc[:,'G_A']
    G_A = G_A.values
    cycle_A = errors.loc[:,'cycle_A']
    cycle_A = cycle_A.values
    
    figA, axA = plt.subplots(3,1,constrained_layout=True)
    figA.suptitle("T1-T2 Loss vs epochs", fontsize=14)
    axA[0].plot(epochs,D_A,color='green')
    axA[0].set_xlabel('Epochs')
    axA[0].set_ylabel('Loss')
    axA[0].set_title('T1 Discriminator - MSE Loss')
    axA[0].set_yticks([0, 0.5, 1, 1.5, 2])
    axA[0].set_ylim(0, 2)

    axA[1].plot(epochs,cycle_A,color='red')
    axA[1].set_xlabel('Epochs')
    axA[1].set_ylabel('Loss')
    axA[1].set_title('T1 Cycle - L1 Loss')
    axA[1].set_yticks([0, 0.5, 1, 1.5, 2])
    axA[1].set_ylim(0, 2)
    
    axA[2].plot(epochs,G_A,color='blue')
    axA[2].set_xlabel('Epochs')
    axA[2].set_ylabel('Loss')
    axA[2].set_title('T1 Generator - MSE Loss')
    axA[2].set_yticks([0, 0.5, 1, 1.5, 2])
    axA[2].set_ylim(0, 2)
    plt.savefig(os.path.join(datasave,'LossVSepoch_T1-T2.pdf'))

    
    D_B = errors.loc[:,'D_B']
    D_B = D_B.values
    G_B = errors.loc[:,'G_B']
    G_B = G_B.values
    cycle_B = errors.loc[:,'cycle_B']
    cycle_B = cycle_B.values
    
    figB, axB = plt.subplots(3,1,constrained_layout=True)
    figB.suptitle("T2-T1 Loss vs epochs", fontsize=14)    
    axB[0].plot(epochs,D_B,color='green')
    axB[0].set_xlabel('Epochs')
    axB[0].set_ylabel('Loss')
    axB[0].set_title('T2 Discriminator - MSE Loss')
    axB[0].set_yticks([0, 0.5, 1, 1.5, 2])
    axB[0].set_ylim(0, 2)


    axB[1].plot(epochs,cycle_B,color='red')
    axB[1].set_xlabel('Epochs')
    axB[1].set_ylabel('Loss')
    axB[1].set_title('T2 Cycle - L1 Loss')
    axB[1].set_yticks([0, 0.5, 1, 1.5, 2])
    axB[1].set_ylim(0, 2)

    
    axB[2].plot(epochs,G_B,color='blue')
    axB[2].set_xlabel('Epochs')
    axB[2].set_ylabel('Loss')
    axB[2].set_title('T2 Generator - MSE Loss')
    axB[2].set_yticks([0, 0.5, 1, 1.5, 2])
    axB[2].set_ylim(0, 2)
    plt.savefig(os.path.join(datasave,'LossVSepoch_T2-T1.pdf'))


    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle("Segmentation Loss vs epochs", fontsize=14)
    ax.set_yticks([1, 0.5, 0])
    ax.set_ylim(0, 1)
    Seg = errors.loc[:,'Seg']
    ax.set_title('Segmentation - Tversky Loss')
    ax.plot(epochs,Seg,color='blue')
    plt.savefig(os.path.join(datasave,'Segmentation_LossVSepoch.pdf'))




    


if __name__=='__main__':
    opt=initialize().parse_args()
    datasave=opt.datasave
    dataroot=opt.dataroot

file1 = open(dataroot, 'r') 
lines = file1.readlines()
dicts = list()
for i, line in enumerate(lines):
    if line[0]=='=':
        continue
    parts = line.split('(')[1].split(' ')
    parts.pop(-1)
    dict_tmp = dict()
    dict_tmp['epoch'] = float(parts[1].split(',')[0])
    dict_tmp['iters'] = float(parts[3].split(',')[0])
    dict_tmp['D_A'] = float(parts[7])
    dict_tmp['G_A'] = float(parts[9])
    dict_tmp['cycle_A'] = float(parts[11])
    dict_tmp['D_B'] = float(parts[13])
    dict_tmp['G_B'] = float(parts[15])
    dict_tmp['cycle_B'] = float(parts[17])
    dict_tmp['Seg'] = float(parts[19])
    dicts.append(dict_tmp)
loss = pd.DataFrame(dicts)

mkdir(datasave)
plot_current_errorsVSepochs(loss,datasave)
plot_current_errorsVSiterations(loss,datasave)



