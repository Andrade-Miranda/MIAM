#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM
pwd

source ~/anaconda3/bin/activate root
conda activate transformers


python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model CNN_h+VIT_n  --mode MeanEnsemb --chkname lastCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter/second_trial --folds 0 1 2 3 4

#python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model MCNN_h+VIT_s  --mode MeanEnsemb --chkname lastCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter/second_trial/MCNN_h+VIT --folds 0 1 2 3 4

#python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model MCNN_h+VIT_cv  --mode MeanEnsemb --chkname lastCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter/second_trial/MCNN_h+VIT --folds 0 1 2 3 4

#python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model SwinTrans3D  --mode MeanEnsemb --chkname lastCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter/second_trial/swinUNETR --folds 0 1 2 3 4

#python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model UNETR  --mode MeanEnsemb --chkname lastCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter/second_trial --folds 0 1 2 3 4

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model SegResNetVAE  --mode MeanEnsemb --chkname lastCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter/second_trial/SegResNetVAE --folds 0 1 2 3 4

