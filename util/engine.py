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
#from util.analysis_utils import calculate_dsc
from util.eval import evaluate
import json
#from scipy.io import savemat
import torch.nn.functional as F
from data.data_loader import CreateDataLoader
from config.train_setup import TrainSetup
from batchgenerators.utilities.file_and_folder_operations import join,maybe_mkdir_p,subfiles,isfile
from util.testing_setup import save_segmentation_nifti_from_softmax,save_segmentation_nifti_softmax
from picai_eval import evaluate_folder
from util import seg_metrics as sg
from util.visualizer import Picai_ResultsPlots
from help_fnct.UncertainSmallEmpty.evaluator import evaluate_folders


from monai.data import (
    decollate_batch
)
from monai.transforms import (
    Compose,
    Activations
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


###################TRaining scheme#############################################################################
def optimize_model(model, optimizer, loss_func,scaler,lr_scheduler, train_gen, args, tracking_metrics, writer,wandb_logger,Config,Debug=None):
    """Optimize model x N training steps per epoch + update learning rate"""

    train_loss, step = 0,  0
    dice_loss,focal_loss=0,0
    trainingKeys=[]
    start_time = time.time()
    epoch = tracking_metrics['epoch']
    num_updates = epoch * args.num_training_steps_per_epoch

    model.train()
    # for each mini-batch or optimization step
    for batch_data in train_gen:
        step += 1
        num_updates += 1
        try:
            inputs = batch_data["image"].to(args.device, non_blocking=True)#image.to(device, non_blocking=True)
            labels = batch_data["label"].to(args.device, non_blocking=True)#label.to(device, non_blocking=True)
        except Exception:
            inputs = torch.from_numpy(batch_data['image']).to(args.device)
            labels = torch.from_numpy(batch_data['label']).to(args.device)
        trainingKeys.append(batch_data['keys'])

        if Debug!=None:
            Debug.segment_thumbnails(image=inputs[0][0:1],label=labels[0],frame_dim=1,savepath="/home/gustavo/test_images/",FigName=trainingKeys[step-1][0])
            

        if args.VAL_AMP:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                #loss,dice_,
                loss = loss_func(outputs, labels)
        else: # full precision
            outputs = model(inputs)
            #loss,dice_,
            loss = loss_func(outputs, labels)   
        train_loss += loss.item()
        #dice_loss+= dice_.item()
        #focal_loss+= focal_.item()

        labels_ = F.one_hot(labels[:, -1, ...].long(), num_classes=args.output_nc).float()
        labels_ = torch.moveaxis(labels_, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
        Config.Config.dice_metricTrain(Config.Config.post_trans(outputs),labels_)

        # backpropagate + optimize
        optimizer.zero_grad()
        # ⭐️ ⭐️ Scale Gradients
        scaler.scale(loss).backward()
        # ⭐️ ⭐️ Update Optimizer
        scaler.step(optimizer)
        scaler.update()
        
        
        if step >= args.num_training_steps_per_epoch: 
            break
    

    # track training metrics
    train_loss /= step
    dice_loss /= step
    focal_loss /= step

    DSCTrain=Config.Config.dice_metricTrain.aggregate().item()
    tracking_metrics['train_loss'] = train_loss
    writer.add_scalar("train_loss", train_loss, epoch+1)

    #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝
    if  args.enable_wandb:
        wandb_logger.log({"train/loss_epoch":train_loss},step=epoch)
        wandb_logger.log({"train/DSC_epoch":DSCTrain},step=epoch)
        
    #  Dice Loss: {dice_loss:.4f}; Focal Loss: {focal_loss:.4f};
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{args.epochs} (Train. Total Loss: {train_loss:.4f}; \
          DSC Train: {DSCTrain:.5f}; \
        Time: {int(time.time()-start_time)}; \
        sec; Steps Completed: {step})", flush=True)
    Config.Config.dice_metricTrain.reset()

    return model, optimizer, train_gen, tracking_metrics, writer,wandb_logger

#############################################################################


##########################VALIDATTION WITH CLASSIFICATION##########################################
def validate_model(model, loss_func,optimizer, valid_gen, args, tracking_metrics, writer,wandb_logger,Config):
    """Validate model per N epoch + export model weights"""
    all_valid_preds, all_valid_labels,all_valid_keys,val_loss = [], [],[], 0
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
        valloss = loss_func(outputs, valid_labels)# tomo el zero para poder hacer one-hot
        val_loss += valloss.item()

        labels = F.one_hot(valid_labels[:, -1, ...].long(), num_classes=2).float()
        labels = torch.moveaxis(labels, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
        Config.Config.dice_metricVal(Config.Config.post_trans(outputs),labels)#one-hot format

        # test-time augmentation
        valid_images = [valid_images, torch.flip(valid_images, [4]).to(args.device)]

        # aggregate all validation predictions
        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        preds = [
             torch.sigmoid(model(x))[:,-1, ...].detach().cpu().numpy()
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
        
        all_valid_labels += [labels[:, -1, ...].cpu().numpy()] #append to the list the validation true label
        #pred_bin=Config.Config.post_trans(outputs)[:, -1, ...]
        #all_valid_preds += #[torch.sigmoid(outputs)[:, -1, ...].detach().cpu().numpy()]
        all_valid_keys += [i for i in valid_data['keys']]

        if step >= args.num_validation_steps_per_epoch: 
            break

    DSC_val=Config.Config.dice_metricVal.aggregate().item()
    # track validation metrics
    start_time = time.time()
    valid_metrics = evaluate(y_det=iter(np.concatenate([x for x in np.array(all_valid_preds)], axis=0)),
                             y_true=iter(np.concatenate([x for x in np.array(all_valid_labels)], axis=0)),
                             subject_list=all_valid_keys,num_parallel_calls=args.max_num_threads,
                             y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])
    print(f"Time evaluation validation: {int(time.time()-start_time)} sec", flush=True)

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
    tracking_metrics['all_valid_metrics_Dice'].append(DSC_val)


    # export train-time + validation metrics as .xlsx sheet
    metricsData = pd.DataFrame(list(zip(tracking_metrics['all_epochs'],
                                        tracking_metrics['all_train_loss'],
                                        tracking_metrics['all_valid_loss'],
                                        tracking_metrics['all_valid_metrics_auroc'],
                                        tracking_metrics['all_valid_metrics_BestROC_THR'],
                                        tracking_metrics['all_valid_metrics_ap'],
                                        tracking_metrics['all_valid_metrics_BestPR_THR'],
                                        tracking_metrics['all_valid_metrics_ranking'],
                                        tracking_metrics['all_valid_metrics_Dice'])),
                               columns=['epoch', 'train_loss','valid_loss','valid_auroc',
                                        'valid_BestROC_THR', 'valid_ap',
                                        'valid_BestPR_THR','valid_ranking','valid_Dice'])

    # create target folder and save exports sheet
    metrics_file = Path(args.out_dir) / "metrics.xlsx"
    metricsData.to_excel(metrics_file, index=False)

    writer.add_scalar("valid_auroc",   valid_metrics.auroc, epoch+1)
    writer.add_scalar("valid_ap",      valid_metrics.AP,    epoch+1)
    writer.add_scalar("valid_ranking", valid_metrics.score, epoch+1)
    writer.add_scalar("val_loss", val_loss/step, epoch+1)
    writer.add_scalar("val_dice", DSC_val, epoch+1)
    
    #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝
    if  args.enable_wandb:
        #pred_notumor=1-np.array(list(valid_metrics.case_pred.values()))
        #prediction=np.stack((pred_notumor,np.array(list(valid_metrics.case_pred.values()))),axis=-1)      
        wandb_logger.log({"val_loss":tracking_metrics['all_valid_loss'][-1],
                          "valid_auroc":valid_metrics.auroc,
                          "valid_ap":valid_metrics.AP,
                          "valid_dice":DSC_val,
                          "valid_ranking":valid_metrics.score,
                          "roc" : wandb.plot.roc_curve([valid_metrics.case_target[s] for s in valid_metrics.subject_list],
                                                        [[1-valid_metrics.case_pred[s],valid_metrics.case_pred[s]] for s in valid_metrics.subject_list],
                                                        labels=['Benign','Malign'],classes_to_plot=1,
                                                         title='ROC Val'),
                          "pr":wandb.plot.pr_curve([valid_metrics.case_target[s] for s in valid_metrics.subject_list], 
                                                   [[1-valid_metrics.case_pred[s],valid_metrics.case_pred[s]] for s in valid_metrics.subject_list],
                                                   labels=['Benign','Malign'],classes_to_plot=1,
                                                   title='Precision vs Recall Val')}) 

    print(f"Valid. Performance [Benign or Indolent PCa (n={num_neg}) \
        vs. csPCa (n={num_pos})]:\nRanking Score = {valid_metrics.score:.3f},\
        AP = {valid_metrics.AP:.3f}, AUROC = {valid_metrics.auroc:.3f}, \
        DSC = {DSC_val:.5f}, Validation Score = {(valid_metrics.score+DSC_val)/2:.5f}", flush=True)
    
    Config.Config.dice_metricVal.reset()

    # store model checkpoint if validation metric improves
    if valid_metrics.score >(valid_metrics.score+DSC_val)/2:#valid_metrics.score > tracking_metrics['best_metric']:
        tracking_metrics['best_metric'] = (valid_metrics.score+(DSC_val))/2 #val_dice[-1]#valid_metrics.score#val_dice/step #valid_metrics.score
        tracking_metrics['best_metric_epoch'] = epoch + 1
        
        weights_file = Path(args.expr_dir) / "BestCHK.pth"

        print(f"Validation Score Improved! Saving New Best Model -> new best score:{tracking_metrics['best_metric']:.3f}, \
              new best Ranking Score:{valid_metrics.score:.3f}, \
               new best DSC:{tracking_metrics['all_valid_metrics_Dice'][-1]:.3f}", 
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

#######################TEST FOR PICAI#########################################""""
def test_Predict_Rank(model,opt,test_loader,datalen):
    torch.backends.cudnn.benchmark = False
    #setting path
    opt.TrainConfig='TestConfig'
    opt.dataset_mode = 'test'

    #######################################
    output_folder = join('./Output/',opt.dataroot,opt.encoder,opt.name,'predictions')
    output_folder_softmax = join('./Output/',opt.dataroot,opt.encoder,opt.name,'Softmax')
    opt.input_folder= join("./nnUNet/data/nnUnet_raw/nnUNet_raw_data",opt.dataroot,"labelsTr")
    y_true_dir=Path("./nnUNet/data/nnUnet_raw/nnUNet_raw_data") / Path(opt.dataroot) / "labelsTr"
    with open(Path("./nnUNet/data/nnUnet_raw/results/overviews/"+opt.dataroot) / f'PI-CAI_val-fold-{opt.fold}.json') as fp:
        valid_json = json.load(fp)
    subject_list=[valid_json['pat_ids'][i]+'_'+valid_json['study_ids'][i] for i in range(len(valid_json['pat_ids']))]
    maybe_mkdir_p(output_folder)
    maybe_mkdir_p(output_folder_softmax)
    ######################################
    
    #####Load best checkpoint#########
    checkpoint = torch.load(join(opt.expr_dir,"BestCHK.pth"),map_location=opt.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['model_state_dict'],strict=False)
    else:
        model.load_state_dict(
            checkpoint,strict=False)
    print("Replace last weights .... LOAD TRAINED BestCHK WEIGHTS")

    testConfig=TrainSetup(opt,model)
    patientsID=[]
    model.eval()
    step=1


    with torch.no_grad():#Context-manager that disabled gradient calculation.

        if not opt.sigmoid:
            post_trans = Compose(
                [Activations(softmax=True)]#[Activations(sigmoid=True)]
            )
        else:
            post_trans = Compose(
                [Activations(sigmoid=True)]
            )

        opt.outputSoft_dir=output_folder_softmax # only to specify output directory
        createPlots=Picai_ResultsPlots(opt,opt.wandb_logger)

        for preprocessed in test_loader:

            val_inputs = preprocessed["image"].to(opt.device, non_blocking=True)
            val_labels = preprocessed["label"].to(opt.device, non_blocking=True)
            out_fname = preprocessed["keys"].item()
            dct=preprocessed["properties"][0]

            #### save sigmoid mask################
            val_outputs = testConfig.Config.inference(val_inputs)#### inference
            val_outputs_seg = testConfig.Config.post_trans(val_outputs[0][-1][None,...])
            val_outputsSoftmax = post_trans(val_outputs[:,-1])

            patientsID.append(out_fname)
            outputpath=join(output_folder,out_fname+'.nii.gz')
            save_segmentation_nifti_from_softmax(val_outputs_seg.detach().cpu(), outputpath,
                                         dct, order=1,
                                         region_class_order= None,
                                         seg_postprogess_fn= None,seg_postprocess_args=None,#testConfig.Config.postprocessing, seg_postprocess_args= {out_fname},
                                         resampled_npz_fname= None,
                                         non_postprocessed_fname= None, force_separate_z= None,
                                         interpolation_order_z= 0, verbose= True,isbrats=False)
                
            # save softmax prediction
            outputpath_softmax=join(output_folder_softmax,out_fname+'.nii.gz')
            save_segmentation_nifti_softmax(val_outputsSoftmax.detach().cpu(), outputpath_softmax,
                                         dct, order=1,
                                         region_class_order= None,
                                         seg_postprogess_fn=None,seg_postprocess_args=None, #testConfig.Config.postprocessing, seg_postprocess_args= {out_fname},
                                         resampled_npz_fname= None,
                                         non_postprocessed_fname= None, force_separate_z= None,
                                         interpolation_order_z= 0, verbose= True,isbrats=False)

            if step==datalen:
                break
            else:
                step+=1

            
           
        del val_outputs,val_outputs_seg

    labels= [i for i in range(opt.output_nc)]
    metricspercase = sg.write_metrics(labels=labels[1:],
                  gdth_path=join("./nnUNet/data/nnUnet_raw/nnUNet_raw_data",opt.dataroot,"labelsTr"),
                  pred_path=output_folder,
                  metrics=['dice','vs', 'hd95','msd','mdsd','nsd','ba','barycentre'],
                  csv_file=output_folder+'/'+"training_metrics.csv")
    ####evaluation using USE evaluator########################""""""""""""""
    #evaluate_folders(
    #    folder_with_gts="/home/gustavo/Data/dataset/picai/dataset_test/val/valF0_label",#join("./nnUNet/data/nnUnet_raw/nnUNet_raw_data",opt.dataroot,"labelsTr"),
    #    folder_with_predictions=output_folder,
    #    th=1,
    #    labels=(0,1),
    #    name='Use_evaluator')
    #######################################################
    
    print("Evaluate segmentation and/or classification performance",flush=True)
    clasif_metrics = evaluate_folder(y_det_dir=Path(output_folder_softmax),
                              y_true_dir=y_true_dir,
                              subject_list=subject_list,
                              y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
                              detection_map_postfixes=[""],
                              label_postfixes=[""],num_parallel_calls=2
                            )


    clasif_metrics.save(Path(output_folder_softmax) / "metrics-val.json")
    print(f"Evaluation of training performance finished for fold {opt.fold}.")
    print(f"Valid. Performance [Benign or Indolent PCa vs. csPCa]:\nRanking Score = {clasif_metrics.score:.3f},\
        AP = {clasif_metrics.AP:.3f}, AUROC = {clasif_metrics.auroc:.3f}", flush=True)
    createPlots.Plot_curves(clasif_metrics,metricspercase)


def Prostate_Tumor_Augmentation(valid_images,testConfig):
    valid_images = [valid_images, torch.flip(valid_images, [4])]
    preds = [torch.sigmoid(testConfig.Config.inference(x))[:,-1, ...].detach().cpu().numpy()
            for x in valid_images
            ]
    # revert horizontally flipped tta image
    preds[1] = np.flip(preds[1], [3])

    # gaussian blur to counteract checkerboard artifacts in
    # predictions from the use of transposed conv. in the U-Net
    all_valid_preds =np.mean([
            gaussian_filter(x, sigma=1.5)
            for x in preds
            ], axis=0)   #append to the list the validation prediction
    return all_valid_preds

















def test_model(model, test_gen,datalen, args, trainConfig,wandb_logger):
    # 🐝 create a wandb table to log input image, ground_truth masks and predictions
    if  args.enable_wandb:
        columns = ["filename", "image", "ground_truth", "prediction"]
        table = wandb.Table(columns=columns)

    """Validate model per N epoch + export model weights"""
    all_valid_preds, all_valid_labels,all_valid_keys,val_dice = [],[],[],[]
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
            
            outputs=trainConfig.Config.inference(valid_images)
            if valid_labels.shape[1]==1:
                valid_labels = F.one_hot(valid_labels[:, 0, ...].long(), num_classes=args.output_nc).float()
                valid_labels = torch.moveaxis(valid_labels, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
            trainConfig.Config.dice_metricTest(trainConfig.Config.post_trans(outputs),valid_labels)#one-hot format
            
        # test-time augmentation
        #    valid_images = [valid_images, torch.flip(valid_images, [4]).to(args.device)]

        # aggregate all validation predictions
        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        #    preds = [
        #        torch.sigmoid(trainConfig.Config.inference(x))[:, 1, ...].detach().cpu().numpy()
        #        for x in valid_images
        #    ]

        # revert horizontally flipped tta image
        #    preds[1] = np.flip(preds[1], [3])

        # gaussian blur to counteract checkerboard artifacts in
        # predictions from the use of transposed conv. in the U-Net
        #    all_valid_preds += [
        #    np.mean([
        #        gaussian_filter(x, sigma=1.5)
        #       for x in preds
        #    ], axis=0)   #append to the list the validation prediction
        #    ]
            all_valid_labels += [valid_labels[:, -1, ...].detach().cpu().numpy()] #append to the list the validation true label
            pred_bin=trainConfig.Config.post_trans(outputs)[:, -1, ...]
            all_valid_preds += [torch.sigmoid(outputs)[:, -1, ...].detach().cpu().numpy()]
            fn = valid_data['keys'][-1]
            all_valid_keys += [fn]

            dscScore=trainConfig.Config.dice_metricTest.get_buffer()[-1]
            last_metrics['Val_Dice'][valid_data['keys'].tolist()[-1]]=str(dscScore)
            val_dice +=[dscScore]
            print(f"Number:{len(all_valid_keys)} CaseID:{all_valid_keys[-1]} DSC:{dscScore.item():.4f}")

            if args.enable_wandb:
                # log last 20 slices of each 3D image
                total_slice=valid_data["image"].shape[2]
                min=total_slice//2-7
                max=total_slice//2+7
                for slice_no in range(min, max):
                    img = valid_data["image"][0, 0, slice_no,:, :]
                    label = valid_data["label"][0, -1, slice_no,:, :]
                    prediction = pred_bin.detach().cpu().numpy()[0, slice_no,:, :]
                # 🐝 Add data to wandb table dynamically    
                    table.add_data(fn, wandb.Image(img), wandb.Image(label), wandb.Image(prediction))

            if step==datalen:
                break
            else:
                step+=1
   
    num_pos = int(np.sum([np.max(x) for x in np.array([x[0] for x in all_valid_labels],dtype=object)]))
    num_neg = int(len([x for x in np.array([x[0] for x in all_valid_labels],dtype=object)]) - num_pos)
     # 🐝
    if  args.enable_wandb:
        # log predictions table to wandb with `val_predictions` as key
        wandb.log({"val_predictions": table})
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

    valid_metrics = evaluate(y_det=iter([y  for y in np.array([x[0] for x in all_valid_preds],dtype=object)]),
                             y_true=iter([y for y in np.array([x[0] for x in all_valid_labels],dtype=object)]),
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
    
    trainConfig.Config.dice_metricTest.reset()
    # export final validation metrics as json file
    metrics_file = Path(args.out_dir) / "finalMetrics.json"
    with open(metrics_file, "w") as f:
          json.dump(last_metrics, f)

    # 🐝
    if  args.enable_wandb:
        wandb_logger.log({
                        "rocTest" : wandb.plot.roc_curve([valid_metrics.case_target[s] for s in valid_metrics.subject_list],
                                                        [[1-valid_metrics.case_pred[s],valid_metrics.case_pred[s]] for s in valid_metrics.subject_list],
                                                        title='ROC Test'),
                        "prTest":wandb.plot.pr_curve([valid_metrics.case_target[s] for s in valid_metrics.subject_list], 
                                                   [[1-valid_metrics.case_pred[s],valid_metrics.case_pred[s]] for s in valid_metrics.subject_list],
                                                   title='Precision vs Recall Test'),
                        "Confusion Matrix Test WB":wandb.plot.confusion_matrix(
                                         y_true=[valid_metrics.case_target[s] for s in valid_metrics.subject_list],
                                        preds=[np.argmax([1-valid_metrics.case_pred[s],valid_metrics.case_pred[s]]) for s in valid_metrics.subject_list],     
                                        class_names=['Benign','Malign']),
                        "Confusion Matrix": wandb.sklearn.plot_confusion_matrix(y_true=[valid_metrics.case_target[s] for s in valid_metrics.subject_list],
                                                                                 y_pred=[np.argmax([1-valid_metrics.case_pred[s],valid_metrics.case_pred[s]]) for s in valid_metrics.subject_list], 
                                                                                 labels=['Benign','Malign'])
                                        }) 
# --------------------------------------------------------------------------------------------------------------------------) 
    print(f"Valid. Performance [Benign or Indolent PCa (n={num_neg}) \
        vs. csPCa (n={num_pos})]:\nRanking Score = {valid_metrics.score:.3f},\
        AP = {valid_metrics.AP:.3f}, AUROC = {valid_metrics.auroc:.3f}, \
        DSC = {float(sum(val_dice)/len(val_dice)):.3f}", flush=True)

    return valid_metrics
