#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.losses import DiceCELoss,DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from monai.optimizers import Novograd
from timm.utils import NativeScaler

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
        self.loss_function = DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        self.optimizer = Novograd(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.loss_scaler=NativeScaler()
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.epochs)

        self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
            )
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_EMAmetric = DiceMetric(include_background=True, reduction="mean")
        
        self.Recall_Precision=ConfusionMatrixMetric(include_background=True,metric_name=('sensitivity','precision'),reduction="mean")
        

    # define inference method
    def inference(self,input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.opt.imageSize,
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5,
                )
        if self.opt.VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
