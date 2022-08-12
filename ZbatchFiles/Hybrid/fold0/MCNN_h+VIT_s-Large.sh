#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=BraTSMCNN_h+VIT_s-LE
#SBATCH --output=/homes/gustavo/Code/CNNTrans/Output/BraTSMCNN_h+VIT_s-LE/BraTSMCNN_h+VIT_s-LE.out
#SBATCH --error=/homes/gustavo/Code/CNNTrans/Output/BraTSMCNN_h+VIT_s-LE/BraTSMCNN_h+VIT_s-LE.err
#BraTSMCNN_h+VIT_s-Lembed

cd /homes/gustavo/Code/CNNTrans
pwd
singularity exec --nv ./Trans.sif python3 -W ignore ./HybridTrans.py --name BraTSMCNN_h+VIT_s-LEF0 --encoder MCNN_h+VIT_s --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --patchSize 1 --batchSize 2 --epochs 300 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --hybrid --filters_Encoder 16 32 64 128 --lr 2e-3 --fold 0

exit


