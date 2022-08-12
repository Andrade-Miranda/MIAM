#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=MCNNl_VITs
#SBATCH --output=/homes/gustavo/Code/CNNTrans/Output/MCNNl_VITs/MCNNl_VITs_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/CNNTrans/Output/MCNNl_VITs/MCNNl_VITs_%A_%a.err
#SBATCH --array=0-4
#VITm

cd /homes/gustavo/Code/CNNTrans
pwd

fold=(0 1 2 3 4)   
name=(MCNNl_VITsF0 MCNNl_VITsF1 MCNNl_VITsF2 MCNNl_VITsF3 MCNNl_VITsF4) 

singularity exec --nv ./Trans.sif python3 -W ignore ./HybridTrans.py --name "${name[$SLURM_ARRAY_TASK_ID]}" --encoder MCNN_l+VIT_s --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --patchSize 4 --batchSize 2 --epochs 150 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 2e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --hybrid

exit


