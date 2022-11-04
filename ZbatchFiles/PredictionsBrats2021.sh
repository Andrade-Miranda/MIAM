#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM
pwd

source ~/anaconda3/bin/activate root
conda activate transformers


python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/BratS2021/predictions/FinalResults --task_name Task004_BraTS2021 --model MCNN_h+VIT_n  --mode MeanEnsemb --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/BratS2021/checkpoints/multicenter --folds 0 1 2 3 4

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model SegResNetVAE  --mode MeanEnsemb --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter --folds 0 1 2 3 4

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/BratS2021/predictions/FinalResults --task_name Task004_BraTS2021 --model CNN_h+VIT_n  --mode MeanEnsemb --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/BratS2021/checkpoints/multicenter --folds 0 1 2 3 4

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/BratS2021/predictions/FinalResults --task_name Task004_BraTS2021 --model SwinTrans3D  --mode MeanEnsemb --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/BratS2021/checkpoints/multicenter --folds 0 1 2 3 4

python predict_simple.py --input_folder ./nnUNet/data/nnUnet_raw/nnUNet_raw_data --output_folder /home/gustavo/Data/results/Hektor2021/predictions/FinalResults --task_name Task003_Hektor --model UNETR  --mode MeanEnsemb --chkname BestCHK.pth --chkdir /home/gustavo/Data/results/Hektor2021/checkpoints/multicenter --folds 0 1 2 3 4



