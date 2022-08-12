#!/bin/bash -l

cd /home/gustavo/Code/Synseg-Net
pwd



#15
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e15 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 15 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e15/valT2/FakeImg' --dataFakeSeg './Output/results_e15/valT2/FakeSeg' --datasaveImg './Output/results_e15/valT2/FakeNiiImg' --datasaveSeg './Output/results_e15/valT2/FakeNiiSeg'



#25
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e25 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 25 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1



singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e25/valT2/FakeImg' --dataFakeSeg './Output/results_e25/valT2/FakeSeg' --datasaveImg './Output/results_e25/valT2/FakeNiiImg' --datasaveSeg './Output/results_e25/valT2/FakeNiiSeg'


#35
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e35 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 35 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e35/valT2/FakeImg' --dataFakeSeg './Output/results_e35/valT2/FakeSeg' --datasaveImg './Output/results_e35/valT2/FakeNiiImg' --datasaveSeg './Output/results_e35/valT2/FakeNiiSeg'


#45
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e45 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 45 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1


singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e45/valT2/FakeImg' --dataFakeSeg './Output/results_e45/valT2/FakeSeg' --datasaveImg './Output/results_e45/valT2/FakeNiiImg' --datasaveSeg './Output/results_e45/valT2/FakeNiiSeg'


#55
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e55 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 55 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e55/valT2/FakeImg' --dataFakeSeg './Output/results_e55/valT2/FakeSeg' --datasaveImg './Output/results_e55/valT2/FakeNiiImg' --datasaveSeg './Output/results_e55/valT2/FakeNiiSeg'


#65
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e65 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 65 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e65/valT2/FakeImg' --dataFakeSeg './Output/results_e65/valT2/FakeSeg' --datasaveImg './Output/results_e65/valT2/FakeNiiImg' --datasaveSeg './Output/results_e65/valT2/FakeNiiSeg'


#75
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e75 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 75 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e75/valT2/FakeImg' --dataFakeSeg './Output/results_e75/valT2/FakeSeg' --datasaveImg './Output/results_e75/valT2/FakeNiiImg' --datasaveSeg './Output/results_e75/valT2/FakeNiiSeg'


#85
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e85 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 85 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e85/valT2/FakeImg' --dataFakeSeg './Output/results_e85/valT2/FakeSeg' --datasaveImg './Output/results_e85/valT2/FakeNiiImg' --datasaveSeg './Output/results_e85/valT2/FakeNiiSeg'


#95
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e95 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 95 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e95/valT2/FakeImg' --dataFakeSeg './Output/results_e95/valT2/FakeSeg' --datasaveImg './Output/results_e95/valT2/FakeNiiImg' --datasaveSeg './Output/results_e95/valT2/FakeNiiSeg'



#105
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e105 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 105 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1


singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e105/valT2/FakeImg' --dataFakeSeg './Output/results_e105/valT2/FakeSeg' --datasaveImg './Output/results_e105/valT2/FakeNiiImg' --datasaveSeg './Output/results_e105/valT2/FakeNiiSeg'



#125
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e125 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 125 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e125/valT2/FakeImg' --dataFakeSeg './Output/results_e125/valT2/FakeSeg' --datasaveImg './Output/results_e125/valT2/FakeNiiImg' --datasaveSeg './Output/results_e125/valT2/FakeNiiSeg'




#145
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e145 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 145 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e145/valT2/FakeImg' --dataFakeSeg './Output/results_e145/valT2/FakeSeg' --datasaveImg './Output/results_e145/valT2/FakeNiiImg' --datasaveSeg './Output/results_e145/valT2/FakeNiiSeg'





#165
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e165 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 165 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e165/valT2/FakeImg' --dataFakeSeg './Output/results_e165/valT2/FakeSeg' --datasaveImg './Output/results_e165/valT2/FakeNiiImg' --datasaveSeg './Output/results_e165/valT2/FakeNiiSeg'



#185
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e185 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 185 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e185/valT2/FakeImg' --dataFakeSeg './Output/results_e185/valT2/FakeSeg' --datasaveImg './Output/results_e185/valT2/FakeNiiImg' --datasaveSeg './Output/results_e185/valT2/FakeNiiSeg'



#200
singularity exec --nv pytorch.sif python train_yh.py --dataroot ./datasets/yh --name G-resnet_9blocks_S-VSUnet19att_L-hardVoxelWeighting_dice_supervision --batchSize 1 --model test_seg --pool_size 50 --no_dropout --yh_run_model TestSeg --dataset_mode yh_test_seg  --input_nc 3 --output_nc 3 --input_nc_seg 3 --output_nc_seg 1 --test_img_dir ./datasets/valT2 --test_seg_output_dir ./Output/results_e200 --which_direction BtoA --checkpoints_dir ./checkpoints --which_epoch 200 --gpu_ids -1 --no_html --which_model_netSeg uNet19Att --which_model_netG resnet_9blocks --display_id -1

singularity exec pytorch.sif python ./Z_Auxiliar/PostProcess.py  --dataOriginal './datasets/Complete_Data/valT2_complet/' --dataFakeImg './Output/results_e200/valT2/FakeImg' --dataFakeSeg './Output/results_e200/valT2/FakeSeg' --datasaveImg './Output/results_e200/valT2/FakeNiiImg' --datasaveSeg './Output/results_e200/valT2/FakeNiiSeg'


#














