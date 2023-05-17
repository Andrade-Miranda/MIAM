#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.inferers import sliding_window_inference
from pathlib import Path
import numpy as np
from util.losses import FocalLoss
import json


from timm.utils import NativeScaler
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import pandas as pd

from monai.transforms import (
        AsDiscrete,
        )

class PICAIConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
    
        # use case-level class balance to deduce required train-time class weights
        # load datasheets
        with open(Path("./nnUNet/data/nnUnet_raw/results/overviews/"+self.opt.dataroot) / f'PI-CAI_train-fold-{self.opt.fold}.json') as fp:
            train_json = json.load(fp)
        with open(Path("./nnUNet/data/nnUnet_raw/results/overviews/"+self.opt.dataroot) / f'PI-CAI_val-fold-{self.opt.fold}.json') as fp:
            valid_json = json.load(fp)
        
        # load paths to images and labels
        train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
        valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]
        self.class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0])-np.sum(train_json['case_label']))]
        self.class_ratio_v = [int(np.sum(valid_json['case_label'])), int(len(valid_data[0])-np.sum(valid_json['case_label']))]
        self.class_weights = (self.class_ratio_t / np.sum(self.class_ratio_t))
        
        # log dataset definition
        print('Dataset Definition:', "-"*80)
        print(f'Fold Number: {self.opt.fold}')
        print('Data Classes:', list(np.unique(train_json['case_label'])))
        print(f'Train-Time Class Weights: {self.class_weights}')
        print(f'Training Samples [-:{self.class_ratio_t[1]};+:{self.class_ratio_t[0]}]: {len(train_data[1])}')
        print(f'Validation Samples [-:{self.class_ratio_v[1]};+:{self.class_ratio_v[0]}]: {len(valid_data[1])}')

        self.LoadConfig()
    
    def name(self):
        return "PICAI config"
    
    def LoadConfig(self):
        
        self.optimizer = create_optimizer(
            self.opt, self.model)
        
        self.loss_scaler = torch.cuda.amp.GradScaler()#NativeScaler() # if args.use_amp is False, this won't be used
        
        self.loss_function = FocalLoss(alpha=self.class_weights[-1], gamma=1).to(self.opt.device)

        if self.opt.sched is not None:
            self.lr_scheduler, _ =create_scheduler(self.opt, self.optimizer)
        else:
            self.lr_scheduler=None

        self.post_trans = AsDiscrete(threshold=0.5)

        print('#Config Training scheme created')
        self.resume_or_restart_training()

    def resume_or_restart_training(self):
        """Resume/restart training, based on whether checkpoint exists"""

        weights_file = Path(self.opt.expr_dir+ "BestCHK.pth")
        metrics_file = Path(self.opt.out_dir+str(self.opt.fold)+"_metrics.xlsx")

        if self.opt.yh_run_model=='Continue' and weights_file.is_file():
            print("Loading Weights From:", weights_file)
            checkpoint = torch.load(weights_file)

            # load weights and optimizer state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.to(self.opt.device)

            # load train-time metrics from interrupted run
            tracking_metrics = {}
            if metrics_file.is_file():

                saved_metrics = pd.read_excel(metrics_file, engine='openpyxl')
                all_epochs = (saved_metrics['epoch'].values).tolist()
                all_valid_metrics_auroc = (saved_metrics['valid_auroc'].values).tolist()
                all_valid_metrics_FPR = (saved_metrics['valid_FPR'].values).tolist()
                all_valid_metrics_TPR = (saved_metrics['valid_TPR'].values).tolist()
                all_valid_metrics_BestROC_THR = (saved_metrics['valid_BestROC_THR'].values).tolist()
                all_valid_metrics_ap = (saved_metrics['valid_ap'].values).tolist()
                all_valid_metrics_precision = (saved_metrics['valid_precision'].values).tolist()
                all_valid_metrics_recall = (saved_metrics['valid_recall'].values).tolist()
                all_valid_metrics_BestPR_THR = (saved_metrics['valid_BestPR_THR'].values).tolist()
                all_valid_metrics_ranking = (saved_metrics['valid_ranking'].values).tolist()
                all_valid_metrics_Dice = (saved_metrics['valid_Dice'].values).tolist()

                tracking_metrics = {
                    'fold_id':                   self.opt.fold,
                    'start_epoch':               checkpoint['epoch'] + 1,  # resume at next epoch
                    'all_epochs':                all_epochs,
                    'all_train_loss':           (saved_metrics['train_loss'].values).tolist(),
                    'all_valid_loss':           (saved_metrics['valid_loss'].values).tolist(),
                    'all_valid_metrics_auroc':   all_valid_metrics_auroc,
                    'all_valid_metrics_FPR':     all_valid_metrics_FPR,
                    'all_valid_metrics_TPR':     all_valid_metrics_TPR,
                    'all_valid_metrics_BestROC_THR': all_valid_metrics_BestROC_THR,                
                    'all_valid_metrics_ap':      all_valid_metrics_ap,
                    'all_valid_metrics_precision': all_valid_metrics_precision,
                    'all_valid_metrics_recall':  all_valid_metrics_recall,
                    'all_valid_metrics_BestPR_THR':all_valid_metrics_BestPR_THR,
                    'all_valid_metrics_ranking': all_valid_metrics_ranking,
                    'all_valid_metrics_Dice':    all_valid_metrics_Dice,
                    'best_metric':               np.max(all_valid_metrics_ranking),
                    'best_metric_epoch':         all_epochs[all_valid_metrics_ranking.index(
                                                np.max(all_valid_metrics_ranking))]}

                print('Previous Record of Metrics Loaded:', metrics_file)
            else:
                print('Previous Record of Metrics Not Found:', metrics_file)

                tracking_metrics = {
                    'fold_id':                    self.opt.fold,
                    'start_epoch':                checkpoint['epoch'],
                    'all_epochs':                 [],
                    'all_train_loss':             [],
                    'all_valid_loss':             [],
                    'all_valid_metrics_auroc':    [],
                    'all_valid_metrics_FPR':      [],
                    'all_valid_metrics_TPR':      [],
                    'all_valid_metrics_BestROC_THR':  [],
                    'all_valid_metrics_ap':       [],
                    'all_valid_metrics_precision':  [],
                    'all_valid_metrics_recall':  [],
                    'all_valid_metrics_BestPR_THR':[],
                    'all_valid_metrics_ranking':  [],
                    'all_valid_metrics_Dice':     [],
                    'best_metric': -1,
                    'best_metric_epoch': -1}

            print("Resume Training: Epoch",  tracking_metrics['start_epoch']+1)
            print("Best Validation Metric:", tracking_metrics['best_metric'],
                            "@ Epoch", tracking_metrics['best_metric_epoch'])
        else:
            tracking_metrics = {
                'fold_id':                    self.opt.fold,
                'start_epoch':                0,
                'all_epochs':                 [],
                'all_train_loss':             [],
                'all_valid_loss':             [],
                'all_valid_metrics_auroc':    [],
                'all_valid_metrics_FPR':      [],
                'all_valid_metrics_TPR':      [],
                'all_valid_metrics_BestROC_THR':  [],
                'all_valid_metrics_ap':       [],
                'all_valid_metrics_precision':  [],
                'all_valid_metrics_recall':  [],
                'all_valid_metrics_BestPR_THR':[],            
                'all_valid_metrics_ranking':  [],
                'all_valid_metrics_Dice':     [],
                'best_metric': -1,
                'best_metric_epoch': -1}

            print("Start Training: Epoch", tracking_metrics['start_epoch']+1)
            print("Start value: "+str(np.round(self.opt.lr, 10)), flush=True)
        self.tracking_metrics=tracking_metrics
        
    # define inference method
    def inference(self,input):
        def _compute(input):
         
            return sliding_window_inference(
                    inputs=input,
                    roi_size=self.opt.imageSize,
                    sw_batch_size=self.opt.Val_batchSize,
                    predictor=self.model,
                    overlap=0.5                    
                    )
                
        if self.opt.VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    def name(self):
        return "PICAIConfig"
