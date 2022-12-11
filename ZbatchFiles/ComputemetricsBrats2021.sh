#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM/Z_Auxiliar
pwd

source ~/anaconda3/bin/activate root
conda activate transformers


python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/MCNN_h+VIT_m_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/MCNN_h+VIT_n_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/nnFormer_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/SegResNetVAE_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/Swinfuse_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/SwinTrans3D_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/Transfuse_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/Unet_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/UNETR_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/VIT_m_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/VIT_s_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/MCNN_h+VIT_s_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/CNN_h+VIT_n_Ensemble 

python ComputeMetrics.py --task Task004_BraTS2021Test --pathPrediction /home/gustavo/Data/results/BratS2021/predictions/FinalResults/MCNN_h_Ensemble
