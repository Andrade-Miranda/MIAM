#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

from timm.utils import NativeScaler
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose)

class TIMMConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
        self.LoadConfig()
    
    def name(self):
        return "Brats config"
    
    def LoadConfig(self):
        
        self.optimizer = create_optimizer(
            self.opt, self.model)
        
        self.loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

        if self.opt.sched is not None:
            self.lr_scheduler, _ =create_scheduler(self.opt, self.optimizer)
        else:
            self.lr_scheduler=None


        self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
            )
        
        #metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")                
        self.loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=True)

        
    # define inference method
    def inference(self,input):
        def _compute(input):
         
            return sliding_window_inference(
                    inputs=input,
                    roi_size=self.opt.imageSize,
                    sw_batch_size=self.opt.Val_batchSize,
                    predictor=self.model,
                    overlap=0.5,
                    )
                
        if self.opt.VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    def name(self):
        return "TIMMConfig"
