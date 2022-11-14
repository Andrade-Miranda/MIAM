#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM
pwd

source ~/anaconda3/bin/activate root
conda activate transformers

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/nfolds2 --task_name Task003_Hektor --model SwinTrans3D  --mode Nfold --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/Low_data_regime/ --folds 20  22  24  26  28 

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/nfolds2 --task_name Task003_Hektor --model CNN_h+VIT_n  --mode Nfold --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/Low_data_regime/ --folds 20  22  24  26  28 

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/nfolds2 --task_name Task003_Hektor --model MCNN_h+VIT_n  --mode Nfold --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/Low_data_regime/ --folds 20  22  24  26  28 

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/nfolds2 --task_name Task003_Hektor --model UNETR  --mode Nfold --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/Low_data_regime/ --folds 20  22  24  26  28 


