#!/bin/bash -l
#SBATCH --partition=titanGPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=MCNNh+VITs
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/MCNNh+VITsBig/MCNNh+VITs_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/MCNNh+VITsBig/MCNNh+VITs_%A_%a.err
#SBATCH --array=0
#VITs

cd /homes/gustavo/Code/Hektor
pwd

name=(MCNN_h+VIT_sBigF0)
fold=(0)

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder MCNN_h+VIT_s --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --patchSize 1 --batchSize 2 --epochs 1500 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 32 64 128 256 --lr 2e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --hybrid --loadsplit

exit


