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
import numpy as np
from pathlib import Path
import pandas as pd

import util.loggings 
import time
import os 
import wandb
from scipy.ndimage import gaussian_filter
from report_guided_annotation import extract_lesion_candidates
from util.analysis_utils import calculate_dsc
from util.eval import evaluate
import json
from scipy.io import savemat




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
def evaluate_(data_loader, model, device, use_amp=False):
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



def optimize_model(model, optimizer, loss_func,scaler,lr_scheduler, train_gen, args, tracking_metrics, writer,wandb_logger):
    """Optimize model x N training steps per epoch + update learning rate"""

    train_loss, step = 0,  0
    start_time = time.time()
    key=[]
    epoch = tracking_metrics['epoch']
    if  args.enable_wandb: 
        wandb_logger.log({"epoch":epoch})
    # for each mini-batch or optimization step
    for batch_data in train_gen:
        step += 1
        try:
            inputs = batch_data["image"].to(args.device, non_blocking=True)#image.to(device, non_blocking=True)
            labels = batch_data["label"].to(args.device, non_blocking=True)#label.to(device, non_blocking=True)
        except Exception:
            inputs = torch.from_numpy(batch_data['image']).to(args.device)
            labels = torch.from_numpy(batch_data['label']).to(args.device)

        if args.VAL_AMP:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_func(outputs, labels[:, 0, ...].long())
        else: # full precision
            outputs = model(inputs)
            loss = loss_func(outputs, labels[:, 0, ...].long())   
        train_loss += loss.item()

        key += [batch_data["keys"]]

        # backpropagate + optimize
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #loss.backward()
        #optimizer.step()
        
        if  args.enable_wandb:
            wandb_logger.log({"train/loss":loss.item()})

        # define each training epoch == 100 steps (note: nnU-Net uses 250 steps)
        if step >= args.num_training_steps_per_epoch: 
            break

    # update learning rate
    if args.sched is not None:
        lr_sched=str(np.round(lr_scheduler._get_lr(epoch)[-1]))
    else:
        lr_sched=str(args.lr)
    print("Learning Rate Updated! New Value: "+lr_sched, flush=True)

    # track training metrics
    train_loss /= step
    tracking_metrics['train_loss'] = train_loss
    writer.add_scalar("train_loss", train_loss, epoch+1)
    
    if  args.enable_wandb:
        #log train_loss averaged over epoch, updated_lr and epoch  to wandb
        wandb_logger.log({"train/loss_epoch":train_loss, 'learning_rate':np.round(np.float16(lr_sched), 10),'epoch':epoch})
    
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{args.epochs} (Train. Loss: {train_loss:.4f}; \
        Time: {int(time.time()-start_time)}sec; Steps Completed: {step})", flush=True)

    return model, optimizer, train_gen, tracking_metrics, writer,wandb_logger


def validate_model(model, loss_func,optimizer,post_trans, valid_gen, args, tracking_metrics, writer,wandb_logger):
    """Validate model per N epoch + export model weights"""
    post_trans = post_trans
    all_valid_preds, all_valid_labels,all_valid_keys,val_loss,val_dice = [], [],[], 0,0
    epoch, f = tracking_metrics['epoch'], tracking_metrics['fold_id']
    step=0
    # for each validation sample
    for valid_data in valid_gen:
        step += 1
        try:
            valid_images = valid_data["image"].to(args.device, non_blocking=True)
            valid_labels = valid_data["label"].to(args.device, non_blocking=True)
        except Exception:
            valid_images = torch.from_numpy(valid_data['image']).to(args.device)
            valid_labels = torch.from_numpy(valid_data['label']).to(args.device)
        
        outputs = model(valid_images)
        valloss = loss_func(outputs, valid_labels[:, 0, ...].long())
        val_loss += valloss.item()
        
        # test-time augmentation
        valid_images = [valid_images, torch.flip(valid_images, [4]).to(args.device)]

        # aggregate all validation predictions
        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        preds = [
            torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy()
            for x in valid_images
        ]

        # revert horizontally flipped tta image
        preds[1] = np.flip(preds[1], [3])

        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        all_valid_preds += [
            np.mean([
                gaussian_filter(x, sigma=1.5)
                for x in preds
            ], axis=0)   #append to the list the validation prediction
        ]
        all_valid_labels += [valid_labels.cpu().numpy()[:, 0, ...]] #append to the list the validation true label
        pred_bin=post_trans(all_valid_preds[-1])
        val_dice+=calculate_dsc(pred_bin.cpu().detach().numpy(),all_valid_labels[-1])
        all_valid_keys += [valid_data['keys']]

        if step >= args.num_validation_steps_per_epoch: 
            break


    # track validation metrics
    valid_metrics = evaluate(y_det=iter(np.concatenate([x for x in np.array(all_valid_preds)], axis=0)),
                             y_true=iter(np.concatenate([x for x in np.array(all_valid_labels)], axis=0)),
                             #subject_list=all_valid_keys,num_parallel_calls=1,
                             y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

    num_pos = int(np.sum([np.max(y) for y in np.concatenate(
        [x for x in np.array(all_valid_labels)], axis=0)]))
    num_neg = int(len(np.concatenate([x for x in
                                      np.array(all_valid_labels)], axis=0)) - num_pos)

    tracking_metrics['all_epochs'].append(epoch+1)
    tracking_metrics['all_train_loss'].append(tracking_metrics['train_loss'])
    
    tracking_metrics['all_valid_metrics_auroc'].append(valid_metrics.auroc)
    tracking_metrics['all_valid_metrics_FPR'].append(valid_metrics.calculate_ROC()['FPR'])
    tracking_metrics['all_valid_metrics_TPR'].append(valid_metrics.calculate_ROC()['TPR'])
    tracking_metrics['all_valid_metrics_BestROC_THR'].append(valid_metrics.calculate_ROC()['Best_THR'])

    tracking_metrics['all_valid_metrics_ap'].append(valid_metrics.AP)
    tracking_metrics['all_valid_metrics_precision'].append(valid_metrics.calculate_precision_recall()['precision'])
    tracking_metrics['all_valid_metrics_recall'].append(valid_metrics.calculate_precision_recall()['recall'])
    tracking_metrics['all_valid_metrics_BestPR_THR'].append(valid_metrics.calculate_precision_recall()['Best_THR'])
    
    tracking_metrics['all_valid_metrics_ranking'].append(valid_metrics.score)
    tracking_metrics['all_valid_loss'].append(val_loss/step)
    tracking_metrics['all_valid_metrics_Dice'].append(val_dice/step)


    # export train-time + validation metrics as .xlsx sheet
    metricsData = pd.DataFrame(list(zip(tracking_metrics['all_epochs'],
                                        tracking_metrics['all_train_loss'],
                                        tracking_metrics['all_valid_loss'],
                                        tracking_metrics['all_valid_metrics_auroc'],
                                        tracking_metrics['all_valid_metrics_FPR'],
                                        tracking_metrics['all_valid_metrics_TPR'],
                                        tracking_metrics['all_valid_metrics_BestROC_THR'],
                                        tracking_metrics['all_valid_metrics_ap'],
                                        tracking_metrics['all_valid_metrics_precision'],
                                        tracking_metrics['all_valid_metrics_recall'],
                                        tracking_metrics['all_valid_metrics_BestPR_THR'],
                                        tracking_metrics['all_valid_metrics_ranking'],
                                        tracking_metrics['all_valid_metrics_Dice'])),
                               columns=['epoch', 'train_loss','valid_loss','valid_auroc','valid_FPR','valid_TPR',
                                        'valid_BestROC_THR', 'valid_ap', 'valid_precision','valid_recall',
                                        'valid_BestPR_THR','valid_ranking','valid_Dice'])

    # create target folder and save exports sheet
    metrics_file = Path(args.out_dir) / "metrics.xlsx"
    metricsData.to_excel(metrics_file, encoding='utf-8', index=False)

    writer.add_scalar("valid_auroc",   valid_metrics.auroc, epoch+1)
    writer.add_scalar("valid_ap",      valid_metrics.AP,    epoch+1)
    writer.add_scalar("valid_ranking", valid_metrics.score, epoch+1)
    writer.add_scalar("val_loss", val_loss/step, epoch+1)
    writer.add_scalar("val_dice", val_dice/step, epoch+1)
    
    if  args.enable_wandb:
        pred_notumor=1-np.array(list(valid_metrics.case_pred.values()))
        prediction=np.stack((pred_notumor,np.array(list(valid_metrics.case_pred.values()))),axis=-1)      
        wandb_logger.log({"val_loss":tracking_metrics['all_valid_loss'][-1],
                          "valid_auroc":valid_metrics.auroc,
                          "valid_ap":valid_metrics.AP,
                          "valid_dice":tracking_metrics['all_valid_metrics_Dice'][-1],
                          "valid_ranking":valid_metrics.score,
                          "roc" : wandb.plot.roc_curve(list(valid_metrics.case_target.values()),prediction ,
                            labels=['no tumor','tumor'],classes_to_plot=1),
                          "pr":wandb.plot.pr_curve(list(valid_metrics.case_target.values()), prediction, 
		                    labels=['no tumor','tumor'],classes_to_plot=1)}) 



    print(f"Valid. Performance [Benign or Indolent PCa (n={num_neg}) \
        vs. csPCa (n={num_pos})]:\nRanking Score = {valid_metrics.score:.3f},\
        AP = {valid_metrics.AP:.3f}, AUROC = {valid_metrics.auroc:.3f}, \
        DSC = {(val_dice/step):.3f}", flush=True)

    # store model checkpoint if validation metric improves
    if valid_metrics.score > tracking_metrics['best_metric']:#valid_metrics.score
        tracking_metrics['best_metric'] = valid_metrics.score
        tracking_metrics['best_metric_epoch'] = epoch + 1
        
        weights_file = Path(args.expr_dir) / "BestCHK.pth"

        print(f"Validation Dice Score Improved! Saving New Best Model -> new best score:{tracking_metrics['best_metric']:.3f}, new best DSC:{tracking_metrics['all_valid_metrics_Dice'][-1]:.3f}", 
              flush=True)# ranking score before change to dice
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, weights_file)
        
    # store last model checkpoint
    if epoch+1==args.epochs:
        weights_filelast = Path(args.expr_dir) / "LastCHK.pth"
        print("Saving Last Model", flush=True)# ranking score before change to dice
        torch.save({
                    'epoch': args.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
            }, weights_filelast)


    return model, optimizer, valid_gen, tracking_metrics, writer,wandb_logger,valid_metrics


def test_model(model, test_gen,datalen, args, trainConfig,wandb_logger):
    """Validate model per N epoch + export model weights"""
    post_trans = trainConfig.Config.post_trans
    all_valid_preds, all_valid_labels,all_valid_pred_mask,all_valid_keys,val_dice = [], [],[],[], []
    last_metrics={}
    last_metrics['Val_Dice']={}
    #args.device='cpu'
    #model.to(args.device)
    #load best model weights
    weights_file = Path(args.expr_dir) / "BestCHK.pth"
    checkpoint = torch.load(weights_file,map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['model_state_dict'],strict=False)
    else:
        model.load_state_dict(
            checkpoint,strict=False)
    print("LOAD BEST TRAINED WEIGHTS....")
    step=1
    # for each validation sample
    model.eval()
    with torch.no_grad():
        for valid_data in test_gen:
            try:
                valid_images = valid_data["image"].to(args.device, non_blocking=True)
                valid_labels = valid_data["label"].to(args.device, non_blocking=True)
            except Exception:
                valid_images = torch.from_numpy(valid_data['image']).to(args.device)
                valid_labels = torch.from_numpy(valid_data['label']).to(args.device)
        # test-time augmentation
        #valid_images = [valid_images, torch.flip(valid_images, [4]).to('cpu')]

        # aggregate all validation predictions
        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        #preds = [
        #    torch.sigmoid(trainConfig.Config.inference(x))[:, 1, ...].detach().numpy()
        #    for x in valid_images
        #]

        # revert horizontally flipped tta image
        #preds[1] = np.flip(preds[1], [3])

        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        #all_valid_preds += [
        #    np.mean([
        #        gaussian_filter(x, sigma=1.5)
        #        for x in preds
        #    ], axis=0)   #append to the list the validation prediction
        #]
            preds=torch.sigmoid(trainConfig.Config.inference(valid_images))[:, 1, ...].detach().cpu().numpy()
            all_valid_preds += [ preds]

            all_valid_labels += [valid_labels.cpu().numpy()[:, 0, ...]] #append to the list the validation true label
            all_valid_pred_mask +=[post_trans(all_valid_preds[-1]).numpy()]
            dscScore=calculate_dsc(all_valid_pred_mask[-1],all_valid_labels[-1])
            last_metrics['Val_Dice'][valid_data['keys'].tolist()[-1]]=str(dscScore)
            val_dice +=[dscScore]
            all_valid_keys += [valid_data['keys'].tolist()[-1]]
            print(len(all_valid_keys),valid_data['keys'].tolist()[-1])

            if step==datalen:
                break
            else:
                step+=1

    # track validation metrics
    valid_metrics = evaluate(y_det=iter([x[0] for x in all_valid_labels]),
                             y_true=iter([x[0] for x in all_valid_labels]),
                             subject_list=all_valid_keys,
                             y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])
    
    
    last_metrics['metrics_auroc']=str(valid_metrics.auroc)
    last_metrics['metrics_FPR']=[str(x) for x in valid_metrics.calculate_ROC()['FPR'].tolist()]
    last_metrics['metrics_TPR']=[str(x) for x in valid_metrics.calculate_ROC()['TPR'].tolist()]
    last_metrics['metrics_BestROC_THR']=str(valid_metrics.calculate_ROC()['Best_THR'])
    last_metrics['metrics_ap']=str(valid_metrics.AP)
    last_metrics['metrics_precision']=[str(x) for x in valid_metrics.calculate_precision_recall()['precision'].tolist()]
    last_metrics['metrics_recall']=[str(x) for x in valid_metrics.calculate_precision_recall()['recall'].tolist()]
    last_metrics['metrics_BestPR_THR']=str(valid_metrics.calculate_precision_recall()['Best_THR'])
    last_metrics['metrics_ranking']=str(valid_metrics.score)
    last_metrics['metrics_Dice']=str(sum(val_dice)/len(val_dice))
    
    # export final validation metrics as json file
    metrics_file = Path(args.out_dir) / "finalMetrics.json"
    with open(metrics_file, "w") as f:
          json.dump(last_metrics, f)

    
    if  args.enable_wandb:
        pred_notumor=1-np.array(list(valid_metrics.case_pred.values()))
        prediction=np.stack((pred_notumor,np.array(list(valid_metrics.case_pred.values()))),axis=-1)      
        wandb_logger.log({
                          "rocTest" : wandb.plot.roc_curve(list(valid_metrics.case_target.values()),prediction ,
                            labels=['no tumor','tumor'],classes_to_plot=1),
                          "prTest":wandb.plot.pr_curve(list(valid_metrics.case_target.values()), prediction, 
		                    labels=['no tumor','tumor'],classes_to_plot=1)})

        # Create a table with the columns to plot
        x=[i for i in range(len(all_valid_keys))]
        data = [[case,x, y] for (case,x,y) in zip(all_valid_keys, x,val_dice)]
        table = wandb.Table(data=data, columns = ["CasesID","ID","DSC"])
        wandb.log({"ValScatter/plot" : wandb.plot.scatter(table, "ID", "DSC",
                                 title="Val cases vs DSC Scatter Plot")})
        
        data = [[case,y] for (case,y) in zip(all_valid_keys,val_dice)]
        table = wandb.Table(data=data, columns = ["CasesID","DSC"])
        wandb.log({"ValBar/plot" : wandb.plot.bar(table, "CasesID", "DSC",
                                 title="Val cases vs DSC bar")})


    #save_matrix = {'all_valid_labels': [np.array(i[0],dtype=object) for i in all_valid_labels], 
    #               'all_valid_pred_mask': [np.array(i[0],dtype=object) for i in all_valid_pred_mask],
    #              'all_valid_preds':[np.array(i[0],dtype=object) for i in all_valid_preds]}
    #savemat(os.path.join(args.out_dir,"validationResults.mat"), save_matrix)

    
    return valid_metrics