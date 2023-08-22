#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM
pwd

singularity exec --nv transformers.sif python3 -W ignore ./main_picai.py \
--dataroot Task2204_picai_extraChWG \
--val_interval 1 \
--num_training_steps_per_epoch 15 \
--num_validation_steps_per_epoch 10 \
--validate_min_epoch 2 \
--yh_run_model Train \
--TrainConfig PICAIConfig  \
--batchSize 1 \
--Val_batchSize 1 \
--epochs 5 \
--gpu_ids 0 \
--amsgrad \
--lr 1e-3 \
--fold 0 \
--Deterministic \
--region None \
--project LocalTrials \
--name example \
--VAL_AMP \
--nameRun example \
--dataset_mode nnUNetExtchan \
--encoder CNN_h+VIT_n \
--patchSize 1 \
--hidden_size 512 \
--num_layers 12 \
--mlp_dim 1536 \
--num_heads 8 \
--input_nc 4 \
--output_nc 1 \
--sched cosine \
--loadsplit nnUNet/splits.json \
--plan nnUNetPlansv2.1_plans_3D.pkl 

