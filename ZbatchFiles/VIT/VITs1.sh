#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=VITs
#SBATCH --output=/homes/gustavo/Code/CNNTrans/Output/VITs/VITs_4.out
#SBATCH --error=/homes/gustavo/Code/CNNTrans/Output/VITs/VITs_4.err
#VITs

cd /homes/gustavo/Code/CNNTrans
pwd


singularity exec --nv ./Trans.sif python3 -W ignore ./HybridTrans.py --name VITsF4 --encoder VIT_s --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseConfig --patchSize 16 --batchSize 2 --epochs 150 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 2e-3 --fold 4

exit


