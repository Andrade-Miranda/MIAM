#!/bin/bash -l

cd /home/gustavo/Code/Hektor/Z_Auxiliar
pwd

source ~/anaconda3/bin/activate root
conda activate transformers

python Hektor2Brats_folderConvert.py  --dataroot /home/gustavo/Data/dataset/hecktor2021_train/Test/hecktor_nii_resampled --datasave /home/gustavo/Data/dataset/hecktor2021_train/Test/ --FoldName HektorTest2021

exit
