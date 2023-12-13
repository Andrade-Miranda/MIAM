
#!/bin/bash -l
cd /home/gustavo/Code/Git_workspace/MIAM
pwd

singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore ./main_picai.py \
--dataroot Task2201_picai \
--imageSize 96 96 96 \
--dataset_mode nnUNet \
--encoder MCNN_h+VIT_s \
--num_training_steps_per_epoch 25 \
--num_validation_steps_per_epoch 10 \
--validate_min_epoch 1 \
--yh_run_model Train \
--TrainConfig PICAIConfig  \
--batchSize 1 \
--Val_batchSize 1 \
--epochs 5 \
--warmup_epochs 10 \
--gpu_ids 0 \
--amsgrad \
--lr 1e-4 \
--opt Adamw \
--fold 0 \
--norm_name instance \
--Deterministic \
--region None \
--input_nc 3 \
--output_nc 2 \
--loadsplit nnUNet/splits.json \
--plan nnUNetPlansv2.1_plans_3D.pkl \
--name tp1 \
--patchSize 1 \
--hidden_size 512 \
--num_layers 4 \
--mlp_dim 4096 \
--num_heads 8 \
--sched warmup_cosine

