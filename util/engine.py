#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 01:05:15 2023

@author: gustavoandrade
"""

import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import sys

import util.loggings 

from monai.data import (
    decollate_batch
)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,trainConfig,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, start_steps=None,
                    log_writer=None,wandb_logger=None, lr_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,args=None):
    model.train(True)
    metric_logger = util.loggings.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', util.loggings.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    

    #for data_iter_step in range(num_training_steps_per_epoch):
    for data_iter_step,batch_data in metric_logger.log_every(num_training_steps_per_epoch,data_loader, print_freq, header):
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        
        samples = batch_data["image"].to(args.device, non_blocking=True)#image.to(device, non_blocking=True)
        targets = batch_data["label"].to(args.device, non_blocking=True)#label.to(device, non_blocking=True)
        #print(batch_data["keys"])

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        # train metrics
        train_outputs = trainConfig.Config.inference(samples)
        train_outputs = [trainConfig.Config.post_trans(i) for i in decollate_batch(train_outputs)]
        trainConfig.Config.dice_metric(y_pred=train_outputs, y=targets)        
        Dice_value = trainConfig.Config.dice_metric.aggregate().item()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(Dice=Dice_value)
 
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=optimizer.param_groups[0]["lr"], head="lr")
            log_writer.update(Train_Dice=Dice_value, head="loss")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                    'Rank-0 Batch Wise/train_loss': loss_value,
                    'Rank-0 Batch Wise/train_lr': optimizer.param_groups[0]["lr"],
                    'Rank-0 Batch Wise/train_Dice': Dice_value
                    }, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}       
          


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = util.loggings.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}