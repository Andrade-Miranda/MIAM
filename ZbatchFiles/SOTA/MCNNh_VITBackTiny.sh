#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=MCNNhbackbone
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/MCNNh_VITBack/MCNNh_VITBackTiny_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/MCNNh_VITBack/MCNNh_VITBackTiny_%A_%a.err
#SBATCH --array=0
#VITm

cd /homes/gustavo/Code/Hektor
pwd

lr=(1e-3)   
name=(MCNNh_VITBackTiny_F0_1e-3) 

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder MCNN_h+VIT-backbone --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --batchSize 2 --epochs 1500 --gpu_ids 0  --VAL_AMP --filters_Encoder 32 64 128 256 --lr "${lr[$SLURM_ARRAY_TASK_ID]}" --fold 0 --region None --input_nc 2 --output_nc 1 --loadsplit --patchSize 1 --num_layers 12 --mlp_dim 768 --num_heads 8

exit


