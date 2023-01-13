#!/bin/bash -l

cd /home/gustavo/Code/CNNTrans/Z_Auxiliar
pwd

source ~/anaconda3/bin/activate root
conda activate transformers

python create_JSONfile.py --root_dir /home/gustavo/Data/dataset/hecktor2021_train/Test/Task03_HektorTest --option 2

exit
