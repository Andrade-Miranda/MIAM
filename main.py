import os
import torch
import numpy as np
import random
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from config.train_setup import TrainSetup
from models.models import create_model
import time
#import util.loggings
from util.engine import train_one_epoch
from timm.utils import ModelEma
from monai.data import (
    decollate_batch
)

from util.util import save_model
from monai.utils import set_determinism

############# Load Options####################################################
opt,root_dir,max_epochs,val_interval,Plots=TrainOptions().parse()

#util.loggings.init_distributed_mode(opt)

if opt.Deterministic:
    seed=opt.seed#+ util.loggings.get_rank()
    set_determinism(seed)
    np.random.seed(seed)
    random.seed(seed)

#################TRAIN###################################"
data_loader = CreateDataLoader(opt)
train_loader,val_loader,datalen = data_loader.load_data()
print('#Datasize = %d: Training:%d   Validation:%d' % (len(data_loader),datalen[0],datalen[1]))
########################################################################################################
    
""" --------load model --------------- """ 
model = create_model(opt)
model.to(opt.device)
trainConfig=TrainSetup(opt,model)
print('#Config Training scheme created')
"""-----------------------------------"""

""" -------- model ema --------------- """ 
model_ema = None
if opt.model_ema:
    # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    model_ema = ModelEma(
        model,
        decay=opt.model_ema_decay,
        device='cpu' if opt.model_ema_force_cpu else '',
        resume='')
    print("Using EMA with decay = %.8f" % opt.model_ema_decay)
model_without_ddp = model
"""-----------------------------------"""
num_training_steps_per_epoch =datalen[0]//opt.batchSize
num_validation_steps_per_epoch =datalen[1]//opt.Val_batchSize  

######Initialize metric variables##########
epoch_loss_values = []
val_loss_values = []
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
metric_values_tumor = [] #DICE
metric_total=[]
total_start = time.time()
########################################


for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10,flush=True)
    print(f"epoch {epoch + 1}/{max_epochs}",flush=True)
    epoch_loss = 0
    step = 0
    if opt.log_writer is not None:
        opt.log_writer.set_step(epoch * num_training_steps_per_epoch * opt.update_freq)
    if opt.wandb_logger:
        opt.wandb_logger.set_steps()
    
    
    train_stats = train_one_epoch(
            model, trainConfig.Config.loss_function, train_loader, trainConfig.Config.optimizer,
            opt.device, epoch, trainConfig.Config.scaler, opt.clip_grad, model_ema, mixup_fn=None,
            log_writer=opt.log_writer, wandb_logger=opt.wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=trainConfig.Config.lr_scheduler, wd_schedule_values=None,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=opt.update_freq,
            use_amp=opt.VAL_AMP
        )
    
    
    for batchIt in range(num_training_steps_per_epoch):
        step_start = time.time()
        step += 1
        batch_data = next(train_loader)
        inputs, labels = (
        batch_data["image"].to(opt.device),
        batch_data["label"].to(opt.device),
        ) 
        
        #print("Data batch: %s, %s" %(batch_data['keys'][0],batch_data['keys'][1]),flush=True)
        trainConfig.Config.optimizer.zero_grad()#initialize optimizer
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = trainConfig.Config.loss_function(outputs, labels)
        trainConfig.Config.scaler.scale(loss).backward()
        trainConfig.Config.scaler.step(trainConfig.Config.optimizer)
        trainConfig.Config.scaler.update()
            
        epoch_loss += loss.item()
            
        print(
                f"{step}/{datalen[0] // opt.batchSize}"
                f", train_loss: {loss.item():.5f}"
                f", step time: {(time.time() - step_start):.5f}",flush=True
                )
    trainConfig.Config.lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.5f} Learning rate: {trainConfig.Config.lr_scheduler.get_last_lr()[0]:.2e}",flush=True)

    if (epoch + 1) % val_interval == 0:
        val_loss_epoch=0
        stepval = 0
        model.eval()
        with torch.no_grad():#Context-manager that disabled gradient calculation.
            for batchIt in range(num_validation_steps_per_epoch):
                stepval += 1
                val_data = next(val_loader)
                val_inputs,val_labels= (
                        val_data["image"].to(opt.device),
                        val_data["label"].to(opt.device))
                val_outputs = trainConfig.Config.inference(val_inputs)
                #val loss
                val_loss = trainConfig.Config.loss_function(val_outputs, val_labels)
                val_loss_epoch += val_loss.item()
                                
                ## val metrics
                val_outputs = [trainConfig.Config.post_trans(i) for i in decollate_batch(val_outputs)]
                trainConfig.Config.dice_metric(y_pred=val_outputs, y=val_labels)
                trainConfig.Config.Recall_Precision(y_pred=val_outputs, y=val_labels)
                trainConfig.Config.HausdorffDis(y_pred=val_outputs, y=val_labels)
                trainConfig.Config.SurfDis(y_pred=val_outputs, y=val_labels)
            
            val_loss_epoch /= stepval
            val_loss_values.append(val_loss_epoch)
            
            metric = trainConfig.Config.dice_metric.aggregate().item()
            metric_values_tumor.append(metric)
            
            trainConfig.Config.dice_metric.reset()


# monitoring only dice metrics
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            best_metrics_epochs_and_time[0].append(best_metric)
            best_metrics_epochs_and_time[1].append(best_metric_epoch)
            best_metrics_epochs_and_time[2].append(time.time() - total_start)
            ####save best model
            save_model(epoch,model,trainConfig.Config.optimizer,loss,trainConfig.Config.scaler,trainConfig.Config.lr_scheduler,metric,os.path.join(root_dir,'BestCHK'+".pth"))
            print("saved new best Dice metric model",flush=True)
            Plots.save_Loss_MetricsHektor(epoch_loss_values,val_loss_values, metric_values_tumor,val_interval)
        print(
            f"current epoch: {epoch + 1} current DICE: {metric:.5f}"
            f"\nbest tumor dice: {best_metric:.5f} "
            f" at epoch: {best_metric_epoch}",flush=True
               )
Plots.save_Loss_MetricsHektor(epoch_loss_values,val_loss_values, metric_values_tumor,val_interval)
save_model(epoch,model,trainConfig.Config.optimizer,loss,trainConfig.Config.scaler,trainConfig.Config.lr_scheduler,metric,os.path.join(root_dir,'lastCHK'+".pth"))
print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.5f}",flush=True)
total_time = time.time() - total_start
print(f"train completed, best_dice: {best_metric:.5f} at epoch: {best_metric_epoch}, total time: {total_time}.")

























