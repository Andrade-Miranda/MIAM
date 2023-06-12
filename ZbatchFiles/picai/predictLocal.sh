
cd /home/gustavo/Code/Git_workspace/MIAM
pwd

'
singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2203_picai_extraChWG \
                	--model Unet__extraCH_WG \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU
                	
singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2203_picai_extraChWG \
                	--model MCNN_h__extraCH_WG \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU
			

singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2203_picai_extraChWG \
                	--model MCNN_h+VIT_s__extraCH_WG \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU
                	
singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2203_picai_extraChWG \
                	--model SwinTrans3D__extraCH_WG \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU
'
                	
singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2201_picai \
                	--model CNN_h+VIT_n__z_16F0 \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU                	

singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2201_picai \
                	--model MCNN_h__z_16F0 \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU                 	

singularity exec --nv /home/gustavo/definitions/TORCH/transformers.sif python3 -W ignore \
			/home/gustavo/Code/Git_workspace/MIAM/predict.py \
			--input_folder /home/gustavo/Data/results/Prostate/picai/F0_predict \
                	--output_folder 1stTrial \
                	--task_name Task2201_picai \
                	--model MCNN_h+VIT_s__z_16F0 \
                	--mode Nfold \
                	--folds 0 \
                	--chkname BestCHK.pth \
                	--GPU    

exit


