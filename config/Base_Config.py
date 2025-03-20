##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,HausdorffDistanceMetric,SurfaceDistanceMetric
from monai.optimizers import Novograd
from torch.nn.modules.loss import _Loss

from util.Multiple_sliding_window_inference import Multi_sliding_window_inference



from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose)

class BaseConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
        self.LoadConfig()

    def LoadConfig(self):
        
        self.optimizer = Novograd(self.model.parameters(), lr=self.opt.lr)#torch.optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)#
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(self.opt.epochs*0.2),eta_min=1e-5)
        self.scaler = torch.cuda.amp.GradScaler()
        self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
            )
        
        #metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")        
        self.Recall_Precision=ConfusionMatrixMetric(include_background=True,metric_name=('recall','precision'),reduction="mean",compute_sample=True)
        self.HausdorffDis=HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction="mean")
        self.SurfDis=SurfaceDistanceMetric(include_background=True, symmetric=False, distance_metric='euclidean', reduction="mean")
        
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
        return "BaseConfig"




