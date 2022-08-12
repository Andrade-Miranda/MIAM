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

torch.backends.cudnn.benchmark = True
############# Load Options##########################
opt = TrainOptions().parse()
opt.imageSize=[int(opt.imageSize[i]) for i in range(len(opt.imageSize))]
opt.filters_Encoder=tuple([int(opt.filters_Encoder[i]) for i in range(len(opt.filters_Encoder))])
if opt.region[0]!='None' and opt.dataroot!='Task001_BraTS2021':
    opt.region=tuple([tuple([int(i) for i in x.split(',')]) if len(x)>1 else (int(x),) for x in opt.region])

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
##########################################################"



#################TRAIN###################################"
if OptionMode == 'Train':
    data_loader = CreateDataLoader(opt)
    train_loader,val_loader,datalen = data_loader.load_data()
    print('#Datasize = %d: Training:%d   Validation:%d' % (len(data_loader),datalen[0],datalen[1]))
    
    """ Multiples GPU """ 
    model = create_model(opt)
    """---------------------"""
    # x=torch.rand(2,4,128,128,128)
    # y=model(x)
    trainConfig=TrainSetup(opt,model)
    print('#Config Training scheme created')
    
    
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tumor = []


    total_start = time.time()
    
    # now if this was a network training you would run epochs like this (remember tr_gen and val_gen generate
    # inifinite examples! Don't do "for batch in tr_gen:"!!!):
    num_batches_per_epoch =datalen[0]//opt.batchSize
    num_validation_batches_per_epoch =datalen[1]//opt.Val_batchSize    
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10,flush=True)
        print(f"epoch {epoch + 1}/{max_epochs}",flush=True)
        model.train()
        epoch_loss = 0
        step = 0
        for batchIt in range(num_batches_per_epoch):
            step_start = time.time()
            step += 1
            batch_data = next(train_loader)
            inputs, labels = (
            batch_data["image"].to(opt.device),
            batch_data["label"].to(opt.device),
            ) 
            
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
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}",flush=True
                )
        trainConfig.Config.lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} Learning rate: {trainConfig.Config.lr_scheduler.get_last_lr()[0]:.4f}",flush=True)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():#Context-manager that disabled gradient calculation.
                for batchIt in range(num_validation_batches_per_epoch):
                    val_data = next(val_loader)
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
                metric_values_tumor.append(metric_tc)

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
                    os.path.join(root_dir,'lastestCHK'+".pth"),
                )
                print("saved new best metric model",flush=True)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tumor: {metric_tc:.4f} "
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}",flush=True
                )
    Plots.save_Loss_MetricsHektor(epoch_loss_values, metric_values, best_metric_epoch, best_metric, metric_values_tumor, val_interval)
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}",flush=True)
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

























