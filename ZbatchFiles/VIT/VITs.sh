#!/bin/bash -l
#SBATCH --partition=titanGPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=VITs
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/VITs/VITs_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/VITs/VITs_%A_%a.err
#SBATCH --array=0-4
#VITs

cd /homes/gustavo/Code/Hektor
pwd

fold=(0 1 2 3 4)   
name=(VITsF0 VITsF1 VITsF2 VITsF3 VITsF4) 

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-main.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder VIT_s --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --patchSize 16 --batchSize 2 --epochs 150 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 2e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --loadsplit

exit


