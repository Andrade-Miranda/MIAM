#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric



from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose)

class TestConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
        self.LoadConfig()

    def LoadConfig(self):
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        
        self.post_trans = Activations(sigmoid=True)


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
        return "Testconfig"
