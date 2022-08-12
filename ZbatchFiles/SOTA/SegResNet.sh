#!/bin/bash -l
#SBATCH --partition=1080GPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=SegResNet
#SBATCH --output=/homes/gustavo/Code/CNNTrans/Output/SegResNet/SegResNet_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/CNNTrans/Output/SegResNet/SegResNet_%A_%a.err
#SBATCH --array=0-4
#SegResNet

cd /homes/gustavo/Code/CNNTrans

pwd


fold=(0 1 2 3 4)   
name=(SegResNetF0 SegResNetF1 SegResNetF2 SegResNetF3 SegResNetF4) 


singularity exec --nv ./Trans.sif python3 -W ignore ./HybridTrans.py --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder SegResNet --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --batchSize 2 --epochs 150 --gpu_ids 0 --VAL_AMP --lr 2e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}"

exit


