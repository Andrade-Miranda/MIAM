#!/bin/bash -l
#SBATCH --partition=1080GPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=UMon
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/Unet/Unet_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/Unet/Unet_%A_%a.err
#SBATCH --array=0-9%3
#VITm

cd /homes/gustavo/Code/Hektor
pwd

fold=(0 1 2 3 4 5 6 7 8 9 10)   
epoch=(1000 500 333 250 200 166 143 125 111 100 91)

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --output_dir ./Output/Unet --encoder Unet --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --batchSize 2 --epochs "${epoch[$SLURM_ARRAY_TASK_ID]}" --gpu_ids 0  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 1e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}"  --region None --input_nc 2 --output_nc 1 --loadsplit

exit


