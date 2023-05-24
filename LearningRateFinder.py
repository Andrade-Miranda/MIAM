import os
import torch
import numpy as np
import random
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from config.train_setup import TrainSetup
from models.models import create_model
from ignite.handlers import FastaiLRFinder

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

lr_finder = FastaiLRFinder()
to_save = {"model": model, "optimizer": trainConfig.Config.optimizer}

with lr_finder.attach(train_loader, to_save=to_save) as trainer_with_lr_finder:
    trainer_with_lr_finder.run(train_loader)

# Get lr_finder results
lr_finder.get_results()

# Plot lr_finder results (requires matplotlib)
lr_finder.plot()

# get lr_finder suggestion for lr
lr_finder.lr_suggestion()

