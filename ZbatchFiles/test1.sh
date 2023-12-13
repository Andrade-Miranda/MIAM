#!/bin/bash -l

cd /home/gustavo/Code/Git_workspace/MIAM
pwd

# Define an array of arguments for the Python script
args=(
  "MCNN_h+VIT_s__FL-BTS-dfSize-B4"
  "MCNN_h+VIT_s__FL-BTS-sz-16x96-B4"
  "nnFormer__FL-sz96x96-B4"
  "MedNeXt__FL-Sz-96x96-B4"
  # Add more arguments as needed  "CNN_h+VIT_n__FL-BTS-sz-96x96-B4" singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif 
)

for arg in "${args[@]}"; do
python3 -W ignore predict.py \
--input_fold_prediction /home/gustavo/Data/dataset/picai/dataset_test/picai_test/images \
--y_true_dir /home/gustavo/Data/dataset/picai/dataset_test/picai_test/labels \
--task_name Task2201_picai \
--model "$arg" \
--mode MeanEnsemb \
--Nfolds 0 1 2 3 4 \
--chkname BestCHK.pth \
--GPU \
--saveSoftmax \
--enable_evaluation \
--sigmoid \
--postprocessing Picai_Postprocessing \
--postpro_dir /home/gustavo/Data/dataset/picai/dataset_test/picai_test/labels-WG \
--enable_wandb \
--project PicaiTest


done


