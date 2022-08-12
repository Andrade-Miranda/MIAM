#!/bin/bash -l
#SBATCH --partition=1080GPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=HekTr800
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/CNN_h+VIT_n/CNNh+VITn_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/CNN_h+VIT_n/CNNh+VITn_%A_%a.err
#SBATCH --array=0-4%2
#VITm

cd /homes/gustavo/Code/Hektor
pwd


name=(CNN_h+VIT_nF0 CNN_h+VIT_nF1 CNN_h+VIT_nF2 CNN_h+VIT_nF3 CNN_h+VIT_nF4)
fold=(0 1 2 3 4 )

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-main.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder CNN_h+VIT_n --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --patchSize 1 --batchSize 2 --epochs 800 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 2e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --hybrid --loadsplit

exit


