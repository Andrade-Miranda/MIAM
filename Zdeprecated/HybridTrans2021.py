#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:57:50 2021

@author: gustavo
"""
import os
import torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from config.train_setup import TrainSetup
from models.models import create_model
from util.visualizer import VisualPlots
import time
from monai.data import (
    decollate_batch
)

from monai.utils import set_determinism

## Load Options
opt = TrainOptions().parse()
opt.imageSize=[int(opt.imageSize[i]) for i in range(len(opt.imageSize))]
opt.filters_Encoder=tuple([int(opt.filters_Encoder[i]) for i in range(len(opt.filters_Encoder))])
OptionMode = opt.yh_run_model # load mode, default=train
root_dir=opt.expr_dir
max_epochs = opt.epochs
val_interval = opt.val_interval
Plots=VisualPlots(opt) #I will use to save some segmentation results. At the moment is only for plot loss curve
if opt.Deterministic:
    set_determinism(seed=0)


"""------------------------------------------------------------"""
# use cpu --gpu_ids -1, GPU --gpu_ids>=0
if len(opt.gpu_ids) == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""------------------------------------------------------------"""


if OptionMode == 'Train':
    data_loader = CreateDataLoader(opt)
    train_loader,val_loader = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#Datasize = %d' % (dataset_size))
    
    """ Multiples GPU """ 
    model = create_model(opt)
    """---------------------"""
    trainConfig=TrainSetup(opt,model)
    print('#Config Training scheme created')
    
    
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(max_epochs):
        torch.backends.cudnn.benchmark = True
        epoch_start = time.time()
        print("-" * 10,flush=True)
        print(f"epoch {epoch + 1}/{max_epochs}",flush=True)
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
            batch_data["image"].to(opt.device),
            batch_data["label"].to(opt.device),
            ) 
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = trainConfig.Config.loss_function(outputs, labels)
            
            trainConfig.Config.optimizer.zero_grad()#initialize optimizer
               
            epoch_loss += loss.item()
            
            is_second_order = hasattr(trainConfig.Config.optimizer, 'is_second_order') and trainConfig.Config.optimizer.is_second_order
            trainConfig.Config.scaler(loss=loss,optimizer=trainConfig.Config.optimizer,clip_grad=opt.clip_grad, clip_mode='norm', parameters=model.parameters(), create_graph=is_second_order)
            
            print(
                f"{step}/{len(train_loader) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}",flush=True
                )
        trainConfig.Config.lr_scheduler.step(epoch)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}",flush=True)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            torch.backends.cudnn.benchmark = False
            with torch.no_grad():#Context-manager that disabled gradient calculation.
                for val_data in val_loader:
                    val_inputs,val_labels= (
                        val_data["image"].to(opt.device),
                        val_data["label"].to(opt.device))
                    val_outputs = trainConfig.Config.inference(val_inputs)
                    val_outputs = [trainConfig.Config.post_trans(i) for i in decollate_batch(val_outputs)]
                    trainConfig.Config.dice_metric(y_pred=val_outputs, y=val_labels)
                    trainConfig.Config.dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = trainConfig.Config.dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = trainConfig.Config.dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                trainConfig.Config.dice_metric.reset()
                trainConfig.Config.dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(root_dir,str(best_metric_epoch)+'epoch_'+str(round(best_metric,2))+".pth"),
                )
                print("saved new best metric model",flush=True)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}",flush=True
                )
            torch.backends.cudnn.benchmark = True
    Plots.save_Loss_Metrics(epoch_loss_values, metric_values, best_metric_epoch, best_metric, metric_values_tc, metric_values_wt, metric_values_et, val_interval)
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}",flush=True)
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

























