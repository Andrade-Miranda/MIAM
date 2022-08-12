#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=MCNN_h+VIT_m-LargeBrATsF3
#SBATCH --output=/homes/gustavo/Code/CNNTrans/Output/BraTSMCNN_h+VIT_m-Large_outF3.txt
#SBATCH --error=/homes/gustavo/Code/CNNTrans/Output/BraTSMCNN_h+VIT_m-Large_errorsF3.txt
#MCNN_h+VIT_n-Large

cd /homes/gustavo/Code/CNNTrans
pwd

singularity exec --nv ./Trans.sif python3 -W ignore ./HybridTrans.py --name BraTSMCNN_h+VIT_m-Largefold3 --encoder MCNN_h+VIT_m --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --patchSize 1 --batchSize 2 --epochs 150 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --hybrid --filters_Encoder 16 32 64 128 --lr 2e-3 --fold 3

exit


