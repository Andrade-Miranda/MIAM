#!/bin/bash -l

cd /home/gustavo/Code/SynsegTransformer
pwd

#source ~/anaconda3/bin/activate root
#conda activate SynSeg-Net

singularity exec pytorch.sif python ./Z_Auxiliar/preprocess.py  --dataroot '/home/gustavo/Code/SynsegTransformer/genkyst/Data_separate/T2' --datasave './datasets/Genkyst_v2/T2/img' --suffix '.nii.gz' --numFiles 100

singularity exec pytorch.sif python ./Z_Auxiliar/preprocess.py  --dataroot '/home/gustavo/Code/SynsegTransformer/genkyst/Data_separate/T1' --datasave './datasets/Genkyst_v2/T1/img' --suffix '.nii.gz' --numFiles 50

singularity exec pytorch.sif python ./Z_Auxiliar/preprocess.py  --dataroot '/home/gustavo/Code/SynsegTransformer/genkyst/Data_separate/Fiesta' --datasave './datasets/Genkyst_v2/Fiesta/img' --suffix '.nii.gz' --numFiles 50

singularity exec pytorch.sif python ./Z_Auxiliar/preprocess.py  --dataroot '/home/gustavo/Code/SynsegTransformer/genkyst/Data_separate/seg/BK' --datasave './datasets/Genkyst_v2/T2/seg' --suffix '.nii.gz' --numFiles 100

exit
