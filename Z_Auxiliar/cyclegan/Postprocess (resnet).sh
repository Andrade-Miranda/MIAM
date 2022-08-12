#!/bin/bash -l

cd /home/gustavo/Code/Synseg-Net
pwd



#185
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-U256_S-VSU256_L-DiceNorm --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 1 --output_nc 1 --input_nc_seg 1 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/G-U256_S-VSU256_L-DiceNorm_e185 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 185 --gpu_ids -1 --no_html --which_model_netSeg unet_256 --which_model_netG unet_256 --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/G-U256_S-VSU256_L-DiceNorm_e185/valT2/FakeImg' --dataFakeSeg './Output/G-U256_S-VSU256_L-DiceNorm_e185/valT2/FakeSeg' --datasaveImg './Output/G-U256_S-VSU256_L-DiceNorm_e185/valT2/FakeNiiImg' --datasaveSeg './Output/G-U256_S-VSU256_L-DiceNorm_e185/valT2/FakeNiiSeg'


#200
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-U256_S-VSU256_L-DiceNorm --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 1 --output_nc 1 --input_nc_seg 1 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/G-U256_S-VSU256_L-DiceNorm_e200 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 200 --gpu_ids -1 --no_html --which_model_netSeg unet_256 --which_model_netG unet_256 --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/G-U256_S-VSU256_L-DiceNorm_e200/valT2/FakeImg' --dataFakeSeg './Output/G-U256_S-VSU256_L-DiceNorm_e200/valT2/FakeSeg' --datasaveImg './Output/G-U256_S-VSU256_L-DiceNorm_e200/valT2/FakeNiiImg' --datasaveSeg './Output/G-U256_S-VSU256_L-DiceNorm_e200/valT2/FakeNiiSeg'

