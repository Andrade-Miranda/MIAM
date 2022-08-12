#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=Swin-2000
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/Swin-UNETR/Swin_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/Swin-UNETR/Swin_%A_%a.err
#SBATCH --array=0-2

cd /homes/gustavo/Code/Hektor
pwd

encoder=(SwinTrans3D SwinTrans3DSimple UNETR)
fold=(0 0 0)   
name=(Swin3DF0 Swin3DSimpleF0 UNETRF0) 
hidden_size=(48 96 768)
patchSize=(2 4 16) 

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder "${encoder[$SLURM_ARRAY_TASK_ID]}" --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --batchSize 2 --epochs 2000 --gpu_ids 0 --hidden_size "${hidden_size[$SLURM_ARRAY_TASK_ID]}" --VAL_AMP --lr 2e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --loadsplit --patchSize "${patchSize[$SLURM_ARRAY_TASK_ID]}"

exit


