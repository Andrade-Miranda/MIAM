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
from timm.utils import get_state_dict,ModelEma
import json

from util.loggings import save_on_master
from monai.utils import set_determinism

from monai.data import (
    decollate_batch
)

############# Load Options####################################################
opt,root_dir,max_epochs,val_interval,Plots=TrainOptions().parse()

#util.loggings.init_distributed_mode(opt)

if opt.Deterministic:
    seed=opt.seed#+ util.loggings.get_rank()
    set_determinism(seed)
    np.random.seed(seed)
    random.seed(seed)

""" --------load Data --------------- """ 
data_loader = CreateDataLoader(opt)
train_loader,val_loader,datalen = data_loader.load_data()
print('#Datasize = %d: Training:%d   Validation:%d' % (len(data_loader),datalen[0],datalen[1]))
num_training_steps_per_epoch =datalen[0]//opt.batchSize
num_validation_steps_per_epoch =datalen[1]//opt.Val_batchSize  
"""-----------------------------------"""
    
""" --------load model and config--------------- """ 
model = create_model(opt)
model.to(opt.device)
n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
opt.num_training_steps_per_epoch=num_training_steps_per_epoch
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

    if opt.log_writer is not None:
        opt.log_writer.set_step(epoch * num_training_steps_per_epoch * opt.update_freq)
    if opt.wandb_logger:
        opt.wandb_logger.set_steps()
    
    
    train_stats = train_one_epoch(
            model, trainConfig.Config.loss_function, train_loader, trainConfig.Config.optimizer,trainConfig,
            epoch, trainConfig.Config.loss_scaler,opt.clip_grad,model_ema=model_ema,mixup_fn=None,start_steps=epoch * num_training_steps_per_epoch,
            log_writer=opt.log_writer, wandb_logger=opt.wandb_logger, lr_schedule_values=trainConfig.Config.lr_scheduler,
            num_training_steps_per_epoch=num_training_steps_per_epoch, 
            update_freq=opt.update_freq,use_amp=opt.VAL_AMP,args=opt
        )
    
    if opt.sched is not None:
        trainConfig.Config.lr_scheduler.step(epoch)

    epoch_loss_values.append(train_stats['loss'])

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
            
            val_loss_epoch /= stepval
            val_loss_values.append(val_loss_epoch)
            
            metric = trainConfig.Config.dice_metric.aggregate().item()
            metric_values_tumor.append(metric)
            trainConfig.Config.dice_metric.reset()
            
            "-----------verify the cases of Model ema and lr_scheduler are None before saving checkpoints------"
            mod_ema= get_state_dict(model_ema) if model_ema is not None else model_ema
            scheduler=trainConfig.Config.lr_scheduler.state_dict() if trainConfig.Config.lr_scheduler is not None else trainConfig.Config.lr_scheduler
            "-------------------------------------------------------------------------------------"
            
# monitoring only dice metrics
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            best_metrics_epochs_and_time[0].append(best_metric)
            best_metrics_epochs_and_time[1].append(best_metric_epoch)
            best_metrics_epochs_and_time[2].append(time.time() - total_start)
            ####save best model
            save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': trainConfig.Config.optimizer.state_dict(),
                        'lr_scheduler': scheduler,
                        'epoch': epoch,
                        'model_ema': mod_ema,
                        'scaler': trainConfig.Config.loss_scaler.state_dict(),
                        #'args': opt
                    }, os.path.join(root_dir,'BestCHK'+".pth"))
            #save_model(epoch,model,trainConfig.Config.optimizer,loss,trainConfig.Config.scaler,trainConfig.Config.lr_scheduler,metric,os.path.join(root_dir,'BestCHK'+".pth"))
            print("saved new best Dice metric model",flush=True)
            Plots.save_Loss_Metrics(epoch_loss_values,val_loss_values, metric_values_tumor,val_interval)
        print(
            f"current epoch: {epoch + 1} current DICE: {metric:.5f}"
            f"\nbest tumor dice: {best_metric:.5f} "
            f" at epoch: {best_metric_epoch}",flush=True
                )
    
    
    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 #**{f'test_{k}': v for k, v in val_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters}
    
    
    if opt.log_writer is not None:
        opt.log_writer.flush()
    with open(os.path.join(opt.out_dir, 'logging', "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")

    if opt.wandb_logger:
        opt.wandb_logger.log_epoch_metrics(log_stats)

if opt.wandb_logger and opt.wandb_ckpt:
    opt.wandb_logger.log_checkpoints()

        
        
Plots.save_Loss_Metrics(epoch_loss_values,val_loss_values, metric_values_tumor,val_interval)
save_on_master({
                 'model': model_without_ddp.state_dict(),
                 'optimizer': trainConfig.Config.optimizer.state_dict(),
                 'lr_scheduler': scheduler,
                 'epoch': epoch,
                 'model_ema': mod_ema,
                 'scaler': trainConfig.Config.loss_scaler.state_dict()
                        #'args': opt,
                    }, os.path.join(root_dir,'LastCHK'+".pth"))
#save_model(epoch,model,trainConfig.Config.optimizer,loss,trainConfig.Config.scaler,trainConfig.Config.lr_scheduler,metric,os.path.join(root_dir,'lastCHK'+".pth"))
print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.5f}",flush=True)
total_time = time.time() - total_start
print(f"train completed, best_dice: {best_metric:.5f} at epoch: {best_metric_epoch}, total time: {total_time}.")

























