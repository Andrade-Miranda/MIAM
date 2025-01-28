import os
import torch
import numpy as np
import random
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from config.train_setup import TrainSetup
from models.models import create_model
from util.engineSeg import optimize_model,validate_model,test_Predict_Rank

from monai.utils import set_determinism


############# Load Options####################################################
opt,root_dir,max_epochs,val_interval,Plots=TrainOptions().parse()


""" --------load Data --------------- """ 
data_loader = CreateDataLoader(opt)
train_loader,val_loader,test_loader,datalen = data_loader.load_data()
print('#Data loader scheme created')  
"""-----------------------------------"""
    
""" --------load model and config--------------- """ 
model = create_model(opt)
trainConfig=TrainSetup(opt,model)
model=trainConfig.Config.model
"""-----------------------------------"""


# for each epoch
for epoch in range(trainConfig.Config.tracking_metrics['start_epoch'], opt.epochs):
    print("-" * 10,flush=True)
    trainConfig.Config.tracking_metrics['epoch'] = epoch
    
    model, trainConfig.Config.optimizer, train_loader, trainConfig.Config.tracking_metrics, opt.wandb_logger  = optimize_model( 
        model=model, 
        optimizer=trainConfig.Config.optimizer, 
        loss_func=trainConfig.Config.loss_function,
        scaler=trainConfig.Config.loss_scaler,
        train_gen=train_loader, 
        args=opt, 
        tracking_metrics=trainConfig.Config.tracking_metrics,
        wandb_logger=opt.wandb_logger,
        Config=trainConfig,
        Debug= Plots if opt.debug else None
        )
    
    ############## learning rate update and setup################
    if opt.sched is not None:
        if opt.sched=='warmup_cosine' or opt.sched=='cosine_anneal':
            trainConfig.Config.lr_scheduler.step()
            lrupdate=trainConfig.Config.optimizer.param_groups[0]['lr']
        elif opt.sched =='poly':
            lrupdate=trainConfig.Config.lr_scheduler.step(epoch + 1)
            trainConfig.Config.optimizer.param_groups[0]['lr']=lrupdate
        print(f"Learning Rate Updated! New Value: {lrupdate:.10}", flush=True)
    else:
        lrupdate=trainConfig.Config.lr_scheduler.step(epoch + 1)
        print(f"Learning Rate fix: {lrupdate:.10}", flush=True)
    # #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝
    if  opt.enable_wandb:
        opt.wandb_logger.log({"lr/epoch":lrupdate},step=epoch)
######################################################################################


    # ----------------------------------------------------------------------------------------------------------------------
    # for each round of validation
    if ((epoch+1) % opt.val_interval == 0) and ((epoch+1) >= opt.validate_min_epoch):

        # validate model per N epochs + export model weights
        model.eval()
        with torch.no_grad():  # no gradient updates during validation
            model, trainConfig.Config.optimizer, val_loader, trainConfig.Config.tracking_metrics,opt.wandb_logger = validate_model(
                model=model, 
                loss_func=trainConfig.Config.loss_function,
                optimizer=trainConfig.Config.optimizer, 
                valid_gen=val_loader, 
                args=opt,
                tracking_metrics=trainConfig.Config.tracking_metrics, 
                wandb_logger=opt.wandb_logger,
                Config=trainConfig,
                )
        Plots.save_Loss_Metrics(trainConfig.Config.tracking_metrics['all_train_loss'],trainConfig.Config.tracking_metrics['all_valid_loss'], trainConfig.Config.tracking_metrics['all_valid_metrics_Dice'], opt.val_interval)

print(
    f"Training Complete! Peak Validation Ranking Score: {trainConfig.Config.tracking_metrics['best_metric']:.4f} "
    f"@ Epoch: {trainConfig.Config.tracking_metrics['best_metric_epoch']}")

test_Predict_Rank(model,opt,test_loader,datalen[1])# modify to perform test simultaniously or activate an option for only test


if  opt.enable_wandb:
    opt.wandb_logger.finish()
# --------------------------------------------------------------------------------------------------------------------------

























