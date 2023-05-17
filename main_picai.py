import os
import torch
import numpy as np
import random
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from config.train_setup import TrainSetup
from models.models import create_model
#import util.loggings
from util.engine import optimize_model,validate_model,test_model
from timm.utils import get_state_dict

from monai.utils import set_determinism
import wandb




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
train_loader,val_loader,test_loader,datalen = data_loader.load_data()
opt.num_validation_steps_per_epoch =datalen[1]//opt.Val_batchSize
print('#Data loader scheme created')  
"""-----------------------------------"""
    
""" --------load model and config--------------- """ 
model = create_model(opt)
n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
trainConfig=TrainSetup(opt,model)
model=trainConfig.Config.model
"""-----------------------------------"""


# for each epoch
for epoch in range(trainConfig.Config.tracking_metrics['start_epoch'], opt.epochs):
    print("-" * 10,flush=True)
    trainConfig.Config.tracking_metrics['epoch'] = epoch
    
    model, trainConfig.Config.optimizer, train_loader, trainConfig.Config.tracking_metrics,opt.log_writer, opt.wandb_logger  = optimize_model( 
        model=model, 
        optimizer=trainConfig.Config.optimizer, 
        loss_func=trainConfig.Config.loss_function,
        scaler=trainConfig.Config.loss_scaler,
        lr_scheduler=trainConfig.Config.lr_scheduler, 
        train_gen=train_loader, 
        args=opt, 
        tracking_metrics=trainConfig.Config.tracking_metrics,
        writer=opt.log_writer,
        wandb_logger=opt.wandb_logger)
    

    # ----------------------------------------------------------------------------------------------------------------------
    # for each round of validation
    if ((epoch+1) % opt.val_interval == 0) and ((epoch+1) >= opt.validate_min_epoch):

        # validate model per N epochs + export model weights
        model.eval()
        with torch.no_grad():  # no gradient updates during validation
            model, trainConfig.Config.optimizer, val_loader, trainConfig.Config.tracking_metrics, opt.log_writer,opt.wandb_logger,
            valid_metrics = validate_model(
                model=model, 
                loss_func=trainConfig.Config.loss_function,
                optimizer=trainConfig.Config.optimizer, 
                post_trans=trainConfig.Config.post_trans,
                valid_gen=val_loader, 
                args=opt,
                tracking_metrics=trainConfig.Config.tracking_metrics, 
                writer=opt.log_writer,
                wandb_logger=opt.wandb_logger
                )

test_metrics=test_model(model,test_loader,datalen[1], opt, trainConfig,opt.wandb_logger)

# ------------------------------Final confusion matrix----------------------------------------
predconf=np.array(list(test_metrics.case_pred.values()))
predconf[predconf>trainConfig.Config.tracking_metrics['all_valid_metrics_BestROC_THR'][-1]]=1
predconf[predconf<=trainConfig.Config.tracking_metrics['all_valid_metrics_BestROC_THR'][-1]]=0
opt.wandb_logger.log({"confusion":wandb.sklearn.plot_confusion_matrix(list(test_metrics.case_target.values()), 
                        predconf)}) 
# --------------------------------------------------------------------------------------------------------------------------
print(
    f"Training Complete! Peak Validation Ranking Score: {trainConfig.Config.tracking_metrics['best_metric']:.4f} "
    f"@ Epoch: {trainConfig.Config.tracking_metrics['best_metric_epoch']}")
opt.log_writer.close()
if  opt.enable_wandb:
    opt.wandb_logger.finish()
# --------------------------------------------------------------------------------------------------------------------------

























