#!/bin/bash -l
#SBATCH --partition=titanGPU
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=MCNNh+VITs
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/test/MCNNh+VITs-EMAFocaLF1.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/test/MCNNh+VITs-EMAFocaLF1.err
#VITs

cd /homes/gustavo/Code/Hektor
pwd


singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./MainEMA.py --dataroot Task002_Hektor --name MCNNh+VITs-EMAFocaLF1 --encoder MCNN_h+VIT_s --yh_run_model Train --dataset_mode nnUNet --TrainConfig TIMMConfig --patchSize 1 --batchSize 2 --epochs 150 --gpu_ids 0 --hidden_size 768 --num_layers 12 --mlp_dim 3072 --num_heads 12  --VAL_AMP --filters_Encoder 16 32 64 128 --lr 2e-3 --fold 1 --region None --input_nc 2 --output_nc 1 --hybrid --loadsplit

exit


