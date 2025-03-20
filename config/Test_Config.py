#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,ConfusionMatrixMetric,HausdorffDistanceMetric,SurfaceDistanceMetric
from util.postprocessing import Picai_Postprocessing

from monai.transforms import (
        Activations,
        AsDiscrete,
        )

class TestConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
        self.LoadConfig()

    def LoadConfig(self):
        #### ----metrics------- 
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.Recall_Precision=ConfusionMatrixMetric(include_background=True,metric_name=('recall','precision'),reduction="mean",compute_sample=True)
        self.HausdorffDis=HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction="mean")
        self.SurfDis=SurfaceDistanceMetric(include_background=True, symmetric=False, distance_metric='euclidean', reduction="mean")
        #### ----metrics------- 
        
        #if self.opt.dataset_mode=='MeanEnsemb' or self.opt.dataset_mode=='MCdropOut':# tengo que usar sigmoid si el output channel es 1 o el usuario especifica sigmoid (Brats dataset - Picai)
        if self.opt.sigmoid or self.opt.output_nc==1:
            self.post_trans = Activations(sigmoid=True)
            self.postLast=AsDiscrete(threshold=0.3)
        else:
            self.post_trans = Activations(softmax=True)
            self.postLast=AsDiscrete(argmax=True)                
        #else:
        #    if self.opt.sigmoid or self.opt.output_nc==1:
        #        self.post_trans = Compose(
        #        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        #    else:
        #        self.post_trans = Compose(
        #        [Activations(softmax=True), AsDiscrete(argmax=True)])                

        

    # define inference method
    def inference(self,input,DeppSuper):
        def _compute(input,DeppSuper=False):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.opt.imageSize,
                sw_batch_size=self.opt.Val_batchSize,
                predictor=self.model,
                DeppSuper=DeppSuper,
                overlap=0.5,
                mode='gaussian'
                )
        if self.opt.VAL_AMP:
            with torch.autocast(device_type=self.opt.device):
                return _compute(input,DeppSuper=False)
        else:
            return _compute(input,DeppSuper=False)
        
    def postprocessing(self,input,mask):
        def _computePostpro(input,mask):
            if self.opt.postprocessing == 'Picai_Postprocessing':
                return Picai_Postprocessing(
                    input=input,
                    prostateMask=mask,
                    postpro_dir=self.opt.postpro_dir
                )
            else:
                self.opt.postprocessing=None

        return _computePostpro(input,mask)

    def name(self):
        return "Testconfig"
