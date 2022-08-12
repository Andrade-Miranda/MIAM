#!/bin/bash -l
#SBATCH --partition=titanGPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=UNETR
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/UNETR/UNETR_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/UNETR/UNETR_%A_%a.err
#SBATCH --array=0-10%5
#UNETR

cd /homes/gustavo/Code/Hektor
pwd

fold=(51 52 53 54 55 56 57 58 59 60 67)
epoch=(2000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000)

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --output_dir ./Output/UNETR --encoder UNETR --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --patchSize 16 --batchSize 2 --epochs "${epoch[$SLURM_ARRAY_TASK_ID]}" --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --lr 1e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --loadsplit

exit


