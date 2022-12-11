#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:01:02 2022

@author: gustavo
"""


            
#import seg_metrics.seg_metrics as sg
import metricsforBrats2021 as sg
import argparse


def initialize():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--task',type=str, default='Task004_BraTS2021Test', help='True values')#brats:'Task004_BraTS2021Test' hektor:'Task003_HektorTest'
    parser.add_argument('--pathPrediction', type=str, default='/home/gustavo/Data/results/BratS2021/predictions/FinalResults/CNN_h+VIT_n_Ensemble', help='predictions') 
    parser.add_argument('--csv_file', type=str, default='metrics.csv', help='file to save metrics') 
    parser.add_argument('--labels', nargs='+', default=[0,1,2,4], help='specify the data to convert') #brats:[0,1,2,4] ->hecktor:[0,1]


    #T1DUALin-src T1DUALout-src
    
    return parser

if __name__=='__main__':
    
    opt = initialize().parse_args()
    pathTrue='/home/gustavo/Code/Git_workspace/MIAM/nnUNet/data/nnUnet_raw/nnUNet_raw_data/'+opt.task+'/labelsTr'
    metrics = sg.write_metrics(labels=opt.labels[1:],  # exclude background
                  gdth_path=pathTrue,
                  pred_path=opt.pathPrediction,
                  csv_file=opt.pathPrediction+'/'+opt.csv_file)
    print(metrics)
