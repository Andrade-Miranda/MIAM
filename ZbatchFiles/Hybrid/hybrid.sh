#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=HekTr2000
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/HybridBig/HybridBig_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/HybridBig/HybridBig_%A_%a.err
#SBATCH --array=0-2
#VITm

cd /homes/gustavo/Code/Hektor
pwd

encoder=(CNN_h+VIT_n MCNN_h+VIT_n SCNN_h+VIT_n)   
name=(CNN_h+VIT_nBigF0 MCNN_h+VIT_nBigF0 SCNN_h+VIT_nBigF0)


singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder "${encoder[$SLURM_ARRAY_TASK_ID]}" --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --patchSize 1 --batchSize 2 --epochs 1500 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 32 64 128 256 --lr 2e-3 --fold 0 --region None --input_nc 2 --output_nc 1 --hybrid --loadsplit

exit


