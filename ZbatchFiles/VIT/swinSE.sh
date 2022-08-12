#!/bin/bash -l
#SBATCH --partition=a6000
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name=SwinSE
#SBATCH --output=/homes/gustavo/Code/Hektor/Output/SwinSE/SwinSE_%A_%a.out
#SBATCH --error=/homes/gustavo/Code/Hektor/Output/SwinSE/SwinSE_%A_%a.err
#SBATCH --array=0
#SwinSE

cd /homes/gustavo/Code/Hektor
pwd

fold=(1)
epoch=(500)

singularity exec --nv /homes/gustavo/Code/CNNTrans/Trans.sif python3 -W ignore ./Hektor-mainMedpyDice.py --dataroot Task002_Hektor --output_dir ./Output/SwinSE  --encoder SwinTrans3DSimple --yh_run_model Train --dataset_mode nnUNet --TrainConfig BaseTrainConfig --patchSize 4 --hidden_size 48 --batchSize 2 --epochs "${epoch[$SLURM_ARRAY_TASK_ID]}" --gpu_ids 0 --VAL_AMP --lr 1e-3 --fold "${fold[$SLURM_ARRAY_TASK_ID]}" --region None --input_nc 2 --output_nc 1 --loadsplit

exit


