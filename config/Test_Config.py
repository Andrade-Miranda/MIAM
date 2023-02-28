#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,HausdorffDistanceMetric,SurfaceDistanceMetric



from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose,AsChannelFirst)

class TestConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
        self.LoadConfig()

    def LoadConfig(self):
        #metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.Recall_Precision=ConfusionMatrixMetric(include_background=True,metric_name=('recall','precision'),reduction="mean",compute_sample=True)
        self.HausdorffDis=HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction="mean")
        self.SurfDis=SurfaceDistanceMetric(include_background=True, symmetric=False, distance_metric='euclidean', reduction="mean")

        
        if self.opt.dataset_mode=='MeanEnsemb':
            self.post_trans = Activations(sigmoid=True)
            self.postLast=AsDiscrete(threshold=0.5)
        else:
            self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

    # define inference method
    def inference(self,input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.opt.imageSize,
                sw_batch_size=self.opt.Val_batchSize,
                predictor=self.model,
                overlap=0.5,
                mode='gaussian'
                )
        if self.opt.VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    def name(self):
        return "Testconfig"
