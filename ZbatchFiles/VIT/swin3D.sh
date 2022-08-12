#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=Swin
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/Swin/Swin_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/Swin/Swin_%A_%a.err
#SBATCH --array=0-10%5
#Swin

cd /homes/gustavo/Code/Hektor
pwd

fold=(51 52 53 54 55 56 57 58 59 60 67)
epoch=(2000 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000)

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --output_dir ./Output/Swin  --encoder SwinTrans3D --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --patchSize 4 --hidden_size 48 --batchSize 2 --epochs "${epoch[$SLURM_ARRAY_TASK_ID]}" --gpu_ids 0 --VAL_AMP --lr 1e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --loadsplit

exit


