#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.losses import FocalLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from pathlib import Path
import numpy as np
from util.losses import FL_and_CE_loss,DiceFocalLoss,GeneralizedDiceFocalLoss,FocalLossBin
import json
from monai.metrics import DiceMetric
#from timm.utils import NativeScaler
from timm.scheduler import create_scheduler_v2,scheduler_kwargs
from timm.optim import create_optimizer
import pandas as pd
from util.lr_scheduler import LinearWarmupCosineAnnealingLR,poly_lr,fix_lr
from util.losses import Supervision_loss_Seg


from monai.transforms import (
        AsDiscrete,
        Activations,
        Compose
        )

class DefaultConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model

        self.LoadConfig()
    
    def name(self):
        return "Default config"
    
    
    def LoadConfig(self):
        
        ###########Optimizer#######################################################################""#######################################################################""
        if self.opt.opt!='sgd':
           self.optimizer = create_optimizer(self.opt, self.model)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.opt.lr, weight_decay=self.opt.weight_decay,momentum=self.opt.momentum,nesterov=True)#self.opt.momentum
        #######################################################################""#######################################################################""

        # Allow Amp to perform casts as required by the opt_level
        self.loss_scaler = torch.GradScaler()#NativeScaler() # if args.use_amp is False, this won't be used

        
        ##################LOSS CONFIGURATION##############################################""
        if self.opt.loss_option=='FL_and_CE':
            self.loss_function = FL_and_CE_loss(fl_kwargs={'alpha':self.opt.class_weights[-1],'size_average':False},
                                               ce_kwargs={'reduction': 'sum'},alpha=self.opt.lambda_Loss[0]).to(self.opt.device)#alpha represent the weight for each loss
        elif self.opt.loss_option=='FocalLossbin':
            self.loss_function=FocalLossBin(alpha=self.opt.class_weights[-1]).to(self.opt.device)
        elif self.opt.loss_option=='FocalLoss':
            self.loss_function = FocalLoss(include_background=True,  # only two classes and keep the same weight as before 
                                        to_onehot_y=False, 
                                         gamma=2.0, 
                                         weight=torch.tensor(self.opt.class_weights),
                                         reduction="sum").to(self.opt.device)
        elif self.opt.loss_option=='DiceFocalLoss':
            self.loss_function=DiceFocalLoss(include_background=True, to_onehot_y=not(self.opt.sigmoid),#False if self.opt.dataroot=='Task2201_picai' else True, 
                                        sigmoid=self.opt.sigmoid,
                                        softmax=not(self.opt.sigmoid), 
                                        other_act=None, 
                                        squared_pred=False, jaccard=False, reduction='mean', smooth_nr=1e-05, 
                                        smooth_dr=1e-05, batch=False, gamma=2.0, focal_weight=self.opt.class_weights[1], 
                                        lambda_dice=self.opt.lambda_Loss[0], lambda_focal=self.opt.lambda_Loss[1])
        elif self.opt.loss_option=='GeneralDiceFocalLoss':
            self.loss_function=GeneralizedDiceFocalLoss(include_background=True, to_onehot_y=not(self.opt.sigmoid),#False if self.opt.dataroot=='Task2201_picai' else True, 
                                        sigmoid=self.opt.sigmoid, 
                                        softmax=not(self.opt.sigmoid),
                                        other_act=None, 
                                        reduction='mean', 
                                        smooth_nr=1e-05, 
                                        smooth_dr=1e-05, batch=False, 
                                        gamma=1.0, 
                                        focal_weight=self.opt.class_weights[1], 
                                        lambda_gdl=self.opt.lambda_Loss[0], lambda_focal=self.opt.lambda_Loss[1])
        else:
            print("Choosing by default DiceCELoss")
            self.loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=not(self.opt.sigmoid), sigmoid=self.opt.sigmoid, softmax=not(self.opt.sigmoid))

        if self.opt.DeepSupervision:
            self.loss_function = Supervision_loss_Seg(criterion=self.loss_function,weights=self.opt.weights_supervision,
                                                      attention=self.opt.attention)
       
    ##################Schedule CONFIGURATION##############################################""
        if self.opt.sched is not None:
            updates_per_epoch = self.opt.num_training_steps_per_epoch 
            if self.opt.sched=="poly":
                self.lr_scheduler= poly_lr(self.opt)
            elif self.opt.sched == "warmup_cosine":
                self.lr_scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=self.opt.warmup_epochs, max_epochs=self.opt.epochs)
            elif self.opt.sched == "cosine_anneal":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.epochs*self.opt.dr)
            else:
                self.lr_scheduler, _ =create_scheduler_v2(self.optimizer,
                                                        **scheduler_kwargs(self.opt),
                                                        updates_per_epoch=updates_per_epoch,
                                                        )
        else:
            self.lr_scheduler=fix_lr(self.opt)
        ################Restart o resume le training##################################
        self.resume_or_restart_training()
        ##############################################################################
        if self.tracking_metrics['start_epoch'] > 0 and self.opt.sched is not None:
            if self.opt.sched_on_updates:
                self.lr_scheduler.step_update(self.tracking_metrics['start_epoch'] * updates_per_epoch)
            else:
                self.lr_scheduler.step(self.tracking_metrics['start_epoch'])


    ################################################################################################################################################


        ###will depend of task
        if self.opt.sigmoid:
            self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold=0.3)]
            )
        elif not self.opt.sigmoid and self.opt.output_nc==1:
            self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold=0.3)]
            )
        elif not self.opt.sigmoid and self.opt.output_nc>1:
            self.post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True,to_onehot=self.opt.output_nc)]) 

        #metrics - some task will need to be modified
        self.dice_metricTrain = DiceMetric(include_background=self.opt.output_nc==1, reduction="mean")
        self.dice_metricVal = DiceMetric(include_background=self.opt.output_nc==1, reduction="mean")
        self.dice_metricTest = DiceMetric(include_background=self.opt.output_nc==1, reduction="mean")

        print('#Config Training scheme created')
        

    def resume_or_restart_training(self):
        """Resume/restart training, based on whether checkpoint exists"""

        weights_file = Path(self.opt.expr_dir, "LastCHK.pth")
        metrics_file = Path(self.opt.out_dir,"metrics.xlsx")

        if self.opt.yh_run_model=='Continue' and weights_file.is_file():
            print("Loading Weights From:", weights_file)
            checkpoint = torch.load(weights_file)

            # load weights and optimizer state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
            self.model.to(self.opt.device)

            # load train-time metrics from interrupted run
            tracking_metrics = {}
            if metrics_file.is_file():

                saved_metrics = pd.read_excel(metrics_file, engine='openpyxl')
                all_epochs = (saved_metrics['epoch'].values).tolist()
                all_valid_metrics_Dice = (saved_metrics['valid_Dice'].values).tolist()

                tracking_metrics = {
                    'fold_id':                   self.opt.fold,
                    'start_epoch':               checkpoint['lr_state_dict']['last_epoch'],  # resume at next epoch
                    'all_epochs':                all_epochs,
                    'all_train_loss':           (saved_metrics['train_loss'].values).tolist(),
                    'all_valid_loss':           (saved_metrics['valid_loss'].values).tolist(),
                    'all_valid_metrics_Dice':    all_valid_metrics_Dice,
                    'best_metric':               np.max(all_valid_metrics_Dice),
                    'best_metric_epoch':         all_epochs[all_valid_metrics_Dice.index(
                                                np.max(all_valid_metrics_Dice))]}

                print('Previous Record of Metrics Loaded:', metrics_file)
            else:
                print('Previous Record of Metrics Not Found:', metrics_file)

                tracking_metrics = {
                    'fold_id':                    self.opt.fold,
                    'start_epoch':                checkpoint['epoch']+1,
                    'all_epochs':                 [],
                    'all_train_loss':             [],
                    'all_valid_loss':             [],
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
            with torch.autocast():
                return _compute(input)
        else:
            return _compute(input)

    def name(self):
        return "Default configuration for segmentation"


