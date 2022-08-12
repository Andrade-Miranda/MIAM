#!/bin/bash -l

cd /home/gustavo/Code/Synseg-Net
pwd



###############################################################################################################################################################################################
#
#singularity exec pytorch.sif python Cyclegan_yh.py --dataroot ./datasets/yh --dataset_mode unaligned --model cycle_gan --yh_run_model Test --name CycleGAN-U256  --batchSize 1 --model cycle_gan --input_nc 1 --output_nc 1 --pool_size 50  --no_dropout  --gpu_ids -1 --which_model_netG unet_256  --display_id -1  --test_seg_output_dir ./Output/results_15 --which_direction BtoA --checkpoints_dir ./checkpoints --gpu_ids -1 --no_html --display_id -1 --test_img_dir ./datasets/valT2 --which_epoch 15

#singularity exec pytorch.sif python ./Z_Auxiliar/Postprocess_cyclegan.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_15/valT2/FakeImg' --datasaveImg './Output/results_15/valT2/FakeNiiImg'


#singularity exec pytorch.sif python Cyclegan_yh.py --dataroot ./datasets/yh --dataset_mode unaligned --model cycle_gan --yh_run_model Test --name CycleGAN-U256  --batchSize 1 --model cycle_gan --input_nc 1 --output_nc 1 --pool_size 50  --no_dropout  --gpu_ids -1 --which_model_netG unet_256  --display_id -1  --test_seg_output_dir ./Output/results_25 --which_direction BtoA --checkpoints_dir ./checkpoints --gpu_ids -1 --no_html --display_id -1 --test_img_dir ./datasets/valT2 --which_epoch 25

#singularity exec pytorch.sif python ./Z_Auxiliar/Postprocess_cyclegan.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_25/valT2/FakeImg' --datasaveImg './Output/results_25/valT2/FakeNiiImg'


#singularity exec pytorch.sif python Cyclegan_yh.py --dataroot ./datasets/yh --dataset_mode unaligned --model cycle_gan --yh_run_model Test --name CycleGAN-U256  --batchSize 1 --model cycle_gan --input_nc 1 --output_nc 1 --pool_size 50  --no_dropout  --gpu_ids -1 --which_model_netG unet_256  --display_id -1  --test_seg_output_dir ./Output/results_35 --which_direction BtoA --checkpoints_dir ./checkpoints --gpu_ids -1 --no_html --display_id -1 --test_img_dir ./datasets/valT2 --which_epoch 35

#singularity exec pytorch.sif python ./Z_Auxiliar/Postprocess_cyclegan.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_35/valT2/FakeImg' --datasaveImg './Output/results_35/valT2/FakeNiiImg'

singularity exec pytorch.sif python Cyclegan_yh.py --dataroot ./datasets/yh --dataset_mode unaligned --model cycle_gan --yh_run_model Test --name CycleGAN-U256  --batchSize 1 --model cycle_gan --input_nc 1 --output_nc 1 --pool_size 50  --no_dropout  --gpu_ids -1 --which_model_netG unet_256  --display_id -1  --test_seg_output_dir ./Output/results_45 --which_direction BtoA --checkpoints_dir ./checkpoints --gpu_ids -1 --no_html --display_id -1 --test_img_dir ./datasets/valT2 --which_epoch 45

singularity exec pytorch.sif python ./Z_Auxiliar/Postprocess_cyclegan.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_45/valT2/FakeImg' --datasaveImg './Output/results_45/valT2/FakeNiiImg'






 


#singularity exec pytorch.sif python Cyclegan_yh.py --dataroot ./datasets/yh --dataset_mode unaligned --model cycle_gan --yh_run_model Test --name CycleGAN-U256  --batchSize 1 --model cycle_gan --input_nc 1 --output_nc 1 --pool_size 50  --no_dropout  --gpu_ids -1 --which_model_netG unet_256  --display_id -1  --test_seg_output_dir ./Output/CycleGAN-U256 --which_direction BtoA --checkpoints_dir ./checkpoints --gpu_ids -1 --no_html --display_id -1 --test_img_dir ./datasets/valT2

#singularity exec pytorch.sif python ./Z_Auxiliar/Postprocess_cyclegan.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/CycleGAN-U256/valT2/FakeImg'  --datasaveImg './Output/CycleGAN-U256/valT2/FakeNiiImg' 


#singularity exec pytorch.sif python Cyclegan_yh.py --dataroot ./datasets/yh --dataset_mode unaligned --model cycle_gan --yh_run_model Test --name CycleGAN-Unet19att  --batchSize 1 --model cycle_gan --input_nc 3 --output_nc 3 --pool_size 50  --no_dropout  --gpu_ids -1 --which_model_netG uNet19Att  --display_id -1  --test_seg_output_dir ./Output/CycleGAN-Unet19att --which_direction BtoA --checkpoints_dir ./checkpoints --gpu_ids -1 --no_html --display_id -1 --test_img_dir ./datasets/valT2

#singularity exec pytorch.sif python ./Z_Auxiliar/Postprocess_cyclegan.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/CycleGAN-Unet19att/valT2/FakeImg'  --datasaveImg './Output/CycleGAN-Unet19att/valT2/FakeNiiImg' 

