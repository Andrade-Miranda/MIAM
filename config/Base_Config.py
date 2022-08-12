#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric
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
        
        self.optimizer = Novograd(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.epochs)
        self.scaler = torch.cuda.amp.GradScaler()
        self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
            )
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_EMAmetric = DiceMetric(include_background=True, reduction="mean")

        self.Recall_Precision=ConfusionMatrixMetric(include_background=True,metric_name=('sensitivity','precision'),reduction="mean")
        
        if self.opt.encoder=='Transfuse' or self.opt.encoder=='Swinfuse':
            self.loss_function=TransfuseDiceCELoss()
        else:
            self.loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

        

    # define inference method
    def inference(self,input):
        def _compute(input):
            if self.opt.encoder!='Transfuse' or self.opt.encoder=='Swinfuse':
                return sliding_window_inference(
                    inputs=input,
                    roi_size=(128, 128, 128),
                    sw_batch_size=self.opt.Val_batchSize,
                    predictor=self.model,
                    overlap=0.5,
                    )
            else:
                return Multi_sliding_window_inference(
                    inputs=input,
                    roi_size=(128, 128, 128),
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
        return "Baseconfig"




class TransfuseDiceCELoss(_Loss):

    def __init__(self):
        super(TransfuseDiceCELoss, self).__init__()
        self.loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        
    def forward(self, input, target):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input[0].shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")
        
        
        # ---- loss function ----
        loss4 = self.loss_function(input[0],target)
        loss3 = self.loss_function(input[1],target)
        loss2 = self.loss_function(input[2],target)

        total_loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4


        return total_loss