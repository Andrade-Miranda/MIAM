#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=ConVnext
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/ConVnext/ConVnext_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/ConVnext/ConVnext_%A_%a.err
#SBATCH --array=0
#VITm

cd /homes/gustavo/Code/Hektor
pwd

fold=(0)   
name=(ConVnextBasetrainConfigF0) 

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder ConVnext-UNet --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --batchSize 2 --epochs 1500 --gpu_ids 0  --VAL_AMP --lr 1e-4 --fold "${fold[$SLURM_ARRAY_TASK_ID]}"  --region None --input_nc 2 --output_nc 1 --loadsplit

exit


