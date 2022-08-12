#!/bin/bash -l

cd /home/gustavo/Code/Hektor/Z_Auxiliar
pwd

source ~/anaconda3/bin/activate root
conda activate nnUnet_env

# only when i need to separate data
bash '/home/gustavo/Code/Hektor/Z_Auxiliar/Separate_data.sh'
bash '/home/gustavo/Code/Hektor/Z_Auxiliar/create_JSONfile.sh'


export nnUNet_raw_data_base="/home/gustavo/Code/Hektor/nnUNet/data/nnUnet_raw"
export nnUNet_preprocessed="/home/gustavo/Code/Hektor/nnUNet/data/nnUnet_preprocessed"
export RESULTS_FOLDER="/home/gustavo/Code/Hektor/nnUNet/data/nnUnet_trained_models"

cd /home/gustavo/nnUNet
nnUNet_convert_decathlon_task -i '/home/gustavo/Data/dataset/hecktor2021_train/Test/Task03_HektorTest' # rename folder to taskid
nnUNet_plan_and_preprocess -t 003

