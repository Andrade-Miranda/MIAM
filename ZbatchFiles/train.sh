#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM
pwd

singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore ./predict.py \
--input_folder /home/gustavo/Data/dataset/picai/dataset_test/val/valF0_images \
--y_true_dir /home/gustavo/Data/dataset/picai/dataset_test/val/valF0_label \
--task_name Task2201_picai \
--model UNETR__32-FL \
--mode Nfold \
--folds 0 \
--chkname BestCHK.pth \
--GPU \
--saveProb


