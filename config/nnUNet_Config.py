#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:45:04 2023

@author: gustavo
"""

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
from nnunet.training.learning_rate.poly_lr import poly_lr


from monai.transforms import (
        Activations,
        AsDiscrete,
        Compose)

class nnUNetConfig():
    
    def initialize(self, opt,model):
        self.opt=opt
        self.model=model
        self.opt.weight_decay = 3e-5 # no a good practice but I re-write WD according to nnUnet
        self.LoadConfig()
        
    def LoadConfig(self):
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.opt.lr, weight_decay=self.opt.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None
        self.scaler = torch.cuda.amp.GradScaler()
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
        return "BaseConfig"




    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.opt.epochs, self.opt.lr, 0.9)
        #self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))
