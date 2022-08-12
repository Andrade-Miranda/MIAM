#!/bin/bash -l
#SBATCH --partition=titanGPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=test
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/test/test_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/test/test_%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodelist=cn40
#VITm

cd /homes/gustavo/Code/Hektor
pwd

fold=(0 0 )
epoch=(20 20 )
encoder=(CNN_h+VIT_n  CNN_h+VIT_n)
embed=(perceptron perceptron)
name=(trial1 trial2)

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --output_dir ./Output/test --encoder "${encoder[$SLURM_ARRAY_TASK_ID]}" --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --patchSize 1 --batchSize 2 --epochs "${epoch[$SLURM_ARRAY_TASK_ID]}" --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 1e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --hybrid --loadsplit --pos_embed "${embed[$SLURM_ARRAY_TASK_ID]}" --checkpoints_dir ./checkpoints/test --name "${name[$SLURM_ARRAY_TASK_ID]}"

exit


