#!/bin/bash -l
#SBATCH --partition=a6000,titanGPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=MCNN_h+VIT_nCL
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/Contrastive/Seg_ContrastiveVit.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/Contrastive/Seg_ContrastiveVit.err
#VITm

cd /homes/gustavo/Code/Hektor
pwd



singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-contrast.py --dataroot Task002_Hektor --name Seg_ContrastiveVit --output_dir ./Output/Contrastive --encoder MCNN_h+VIT_n-CL --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig+ContrasLearn --patchSize 1 --batchSize 8 --epochs 200 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 1e-3 --fold 9 --region None --input_nc 2 --output_nc 1 --hybrid --loadsplit --PatchNCELoss


exit


