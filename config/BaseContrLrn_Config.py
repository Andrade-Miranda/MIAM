#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:40:28 2021

@author: gustavo
"""
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric,ConfusionMatrixMetric,HausdorffDistanceMetric,SurfaceDistanceMetric
from monai.optimizers import Novograd
from torch.nn.modules.loss import _Loss
from util.Multiple_sliding_window_inference import Multi_sliding_window_inference
from util.losses import SupConLoss,PatchNCELoss

#from torch.nn import functional as F

#from monai.utils import deprecated_arg
from itertools import combinations


from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose)

class BaseContrLrnConfig():
    
    def initialize(self, opt,model,Cl=[True,False]):
        self.opt=opt
        self.contrastive=Cl
        self.model=model
        self.LoadConfig()

    def LoadConfig(self):
        
        self.optimizer = Novograd(self.model.parameters(), lr=self.opt.lr)#torch.optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(self.opt.epochs*0.2),eta_min=1e-5)
        self.scaler = torch.cuda.amp.GradScaler()
        self.post_trans = Compose(
                [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
            )
        
        #metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.Recall_Precision=ConfusionMatrixMetric(include_background=True,metric_name=('recall','precision'),reduction="mean",compute_sample=True)
        self.HausdorffDis=HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction="mean")
        self.SurfDis=SurfaceDistanceMetric(include_background=True, symmetric=False, distance_metric='euclidean', reduction="mean")
        
        #loss
        self.loss_function = Contrastive_DiceCELoss(self.opt,self.contrastive)
        
    # define inference method
    def inference(self,input):
        def _compute(input):
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
        return "ContrastiveConfig"



class Contrastive_DiceCELoss(_Loss):

    def __init__(self,opt,contrastive):
        super(Contrastive_DiceCELoss,self).__init__()
        self.opt=opt
        self.contrastive=contrastive
                
        self.loss_seg = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=True)
        self.ContrasCONV=SupConLoss()
        self.contrasViT=PatchNCELoss(opt)
        
        
        
        
    def forward(self,zConv,zVit,input,target,targetModality):
        """
        Args:
  
        Raises:
 
        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        losSeg = self.loss_seg(input,target)
        
        zCONV=torch.cat(zConv,dim=0)        
        lossContrCNN=self.ContrasCONV(zCONV,targetModality)*self.opt.lambdaCNN
        
        #lossContrViT=self.contrasViT(zVit[0].view(self.opt.batchSize*zVit[0].shape[1],-1),zVit[1].view(self.opt.batchSize*zVit[1].shape[1],-1))
        lossContrViT=self.contrasViT(zVit[0],zVit[1])*self.opt.lambdaViT

        
        if self.opt.SupContrast and self.opt.PatchNCELoss:
            totalLoss=losSeg+lossContrCNN+lossContrViT
        elif self.opt.SupContrast:
            totalLoss=losSeg+lossContrCNN
        elif self.opt.PatchNCELoss:
            totalLoss=losSeg+lossContrViT
        else:
            totalLoss=losSeg

        return lossContrCNN,lossContrViT,losSeg,totalLoss
    

    
    
    
    
    
    
    
