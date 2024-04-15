#!/bin/bash -l

cd /home/gustavo/Code/Hektor/Z_Auxiliar
pwd

source ~/anaconda3/bin/activate root
conda activate transformers-gpu

python Separate_data.py --dataroot /home/gustavo/Data/dataset/hecktor2021_train/Test/HektorTest2021 --datasave /home/gustavo/Data/dataset/hecktor2021_train/Test/Task03_HektorTest --selectedIds --option Train

python Separate_data.py --dataroot /home/gustavo/Data/dataset/hecktor2021_train/Test/HektorTest2021 --datasave /home/gustavo/Data/dataset/hecktor2021_train/Test/Task03_HektorTest --selectedIds --option test

exit
