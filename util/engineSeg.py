#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 01:05:15 2023

@author: gustavoandrade
"""

import torch
from pathlib import Path
import pandas as pd

import time
import os 

from config.train_setup import TrainSetup
from batchgenerators.utilities.file_and_folder_operations import join,maybe_mkdir_p
from util.testing_setup import save_segmentation_nifti_from_softmax
from nnUNet.nnunet.postprocessing.connected_components import apply_postprocessing_to_folder
from util.eval import UseEvaluator_function


###################TRaining scheme#############################################################################
def optimize_model(model, optimizer, loss_func,scaler, train_gen, args, tracking_metrics,wandb_logger,Config,Debug=None):
    """Optimize model x N training steps per epoch + update learning rate"""

    train_loss, step = 0,  0
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

        if Debug:
            maybe_mkdir_p(os.path.join(args.out_dir,"Debug"))
            Debug.segment_thumbnails(image=inputs[0][0:1],label=labels[0],frame_dim=1,savepath=os.path.join(args.out_dir,"Debug"),FigName=trainingKeys[step-1][0])

        if args.VAL_AMP:
            with torch.autocast(device_type=args.device):
                outputs = model(inputs,args.DeepSupervision)
        else: # full precision
            outputs = model(inputs,args.DeepSupervision)

        loss = loss_func(outputs, labels)   
        train_loss += loss.item()
        ##############################################
        if args.DeepSupervision:
            Config.Config.dice_metricTrain(Config.Config.post_trans(outputs[-1]),labels)
        else:
            Config.Config.dice_metricTrain(Config.Config.post_trans(outputs),labels)


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

    DSCTrain=Config.Config.dice_metricTrain.aggregate().item()
    tracking_metrics['train_loss'] = train_loss

    #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝
    if  args.enable_wandb:
        wandb_logger.log({"train/loss_epoch":train_loss},step=epoch)
        wandb_logger.log({"train/DSC_epoch":DSCTrain},step=epoch)
        
    #  Dice Loss: {dice_loss:.4f}; Focal Loss: {focal_loss:.4f};
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{args.epochs} (Train. Total Loss: {train_loss:.4f}; \
          DSC Train: {DSCTrain:.5f}; \
        Time: {int(time.time()-start_time)}sec; \
        Steps Completed: {step})", flush=True)
    Config.Config.dice_metricTrain.reset()

    return model, optimizer, train_gen, tracking_metrics,wandb_logger
#############################################################################



##########################VALIDATTION ##########################################
def validate_model(model, loss_func,optimizer, valid_gen, args, tracking_metrics,wandb_logger,Config):
    """Validate model per N epoch + export model weights"""
    val_loss = 0
    epoch, f = tracking_metrics['epoch'], tracking_metrics['fold_id']
    step=0
    start_timeVal = time.time()

    # for each validation sample
    for valid_data in valid_gen:
        step += 1
        try:
            valid_images = valid_data["image"].to(args.device, non_blocking=True)
            valid_labels = valid_data["label"].to(args.device, non_blocking=True)
        except Exception:
            valid_images = torch.from_numpy(valid_data['image']).to(args.device)
            valid_labels = torch.from_numpy(valid_data['label']).to(args.device)

        outputs = model(valid_images,args.DeepSupervision)
        valloss = loss_func(outputs, valid_labels)
        val_loss += valloss.item()

        if args.DeepSupervision:
             Config.Config.dice_metricVal(Config.Config.post_trans(outputs[-1]),valid_labels)
        else:
             Config.Config.dice_metricVal(Config.Config.post_trans(outputs),valid_labels)

        if step >= args.num_validation_steps_per_epoch: 
            break

    DSC_val=Config.Config.dice_metricVal.aggregate().item()
    # track validation metrics
    print(f"Time evaluation validation: {int(time.time()-start_timeVal)} sec", flush=True)

    tracking_metrics['all_epochs'].append(epoch+1)
    tracking_metrics['all_train_loss'].append(tracking_metrics['train_loss'])
    
    tracking_metrics['all_valid_loss'].append(val_loss/step)
    tracking_metrics['all_valid_metrics_Dice'].append(DSC_val)


    # export train-time + validation metrics as .xlsx sheet
    metricsData = pd.DataFrame(list(zip(tracking_metrics['all_epochs'],
                                        tracking_metrics['all_train_loss'],
                                        tracking_metrics['all_valid_loss'],
                                        tracking_metrics['all_valid_metrics_Dice'])),
                               columns=['epoch', 'train_loss','valid_loss','valid_Dice'])

    # create target folder and save exports sheet
    metrics_file = Path(args.out_dir) / "metrics.xlsx"
    metricsData.to_excel(metrics_file, index=False)
    
    #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝
    if  args.enable_wandb:
        #pred_notumor=1-np.array(list(valid_metrics.case_pred.values()))
        #prediction=np.stack((pred_notumor,np.array(list(valid_metrics.case_pred.values()))),axis=-1)      
        wandb_logger.log({"val/loss":tracking_metrics['all_valid_loss'][-1],
                          "val/dice":DSC_val})


    print(f"Valid. Performance DSC = {DSC_val:.5f}, Validation Loss = {tracking_metrics['all_valid_loss'][-1]:.5f}", flush=True)
    
    Config.Config.dice_metricVal.reset()

    # store model checkpoint if validation metric improves
    if tracking_metrics['all_valid_metrics_Dice'][-1] > tracking_metrics['best_metric']:
        tracking_metrics['best_metric'] = DSC_val #val_dice[-1]#valid_metrics.score#val_dice/step #valid_metrics.score
        tracking_metrics['best_metric_epoch'] = epoch + 1
        
        weights_file = Path(args.expr_dir) / "BestCHK.pth"

        print(f"Validation Score Improved! Saving New Best Model -> new best DSC:{tracking_metrics['all_valid_metrics_Dice'][-1]:.3f}", 
              flush=True)# ranking score before change to dice
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_state_dict': Config.Config.lr_scheduler.state_dict()
            }, weights_file)
        
        
    # I will save every 1 epochs -> store last model checkpoint 
    if (epoch+1)%args.val_interval==0:
        weights_filelast = Path(args.expr_dir) / "LastCHK.pth"
        print("Saving Last Model", flush=True)# ranking score before change to dice
        torch.save({
                    'epoch': args.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_state_dict': Config.Config.lr_scheduler.state_dict(),
            }, weights_filelast)


    return model, optimizer, valid_gen, tracking_metrics,wandb_logger


#######################TESTING FOR SEG#########################################""""
def test_Predict_Rank(model,opt,test_loader,datalen):
    torch.backends.cudnn.benchmark = False
    #setting to text config mode
    opt.TrainConfig='TestConfig'
    opt.dataset_mode = 'test'

    #######################################
    output_folder = join('./Output/',opt.dataroot,opt.encoder,opt.name,'predictions')
    opt.input_folder= join("./nnUNet/data/nnUnet_raw/nnUNet_raw_data",opt.dataroot,"imagesTr")
    maybe_mkdir_p(output_folder)
    ######################################
    
    #####Load best checkpoint#########
    checkpoint = torch.load(join(opt.expr_dir,"BestCHK.pth"),map_location=opt.device,weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(
            checkpoint['model_state_dict'],strict=False)
    else:
        model.load_state_dict(
            checkpoint,strict=False)
    print("Replace last weights with best checkpoints .... LOAD TRAINED BestCHK WEIGHTS")

    testConfig=TrainSetup(opt,model)
    patientsID=[]
    model.eval()
    step=1


    with torch.no_grad():#Context-manager that disabled gradient calculation.


        for preprocessed in test_loader:

            val_inputs = preprocessed["image"].to(opt.device, non_blocking=True)
            val_labels = preprocessed["label"].to(opt.device, non_blocking=True)
            out_fname = preprocessed["keys"].item()
            dct=preprocessed["properties"][0]

            #### save sigmoid mask################            
            val_outputs = testConfig.Config.inference(val_inputs,DeppSuper=False)#### inference
            val_outputsSoftmax = testConfig.Config.post_trans(val_outputs[:,-1])
            val_outputs_seg = testConfig.Config.postLast(val_outputsSoftmax)

            patientsID.append(out_fname+'.nii.gz')
            outputpath=join(output_folder,out_fname+'.nii.gz')
            ###### setting of the different postprocessing inputs#################
            if opt.postprocessing=='Picai_Postprocessing':
                seg_postprocess_args={outputpath}
            else:
                seg_postprocess_args=None
            
            save_segmentation_nifti_from_softmax(val_outputs_seg.detach().cpu(), outputpath,
                                            dct, order=1,
                                            region_class_order= None,
                                            seg_postprogess_fn= testConfig.Config.postprocessing,
                                            seg_postprocess_args=seg_postprocess_args,#testConfig.Config.postprocessing, seg_postprocess_args= {out_fname},
                                            resampled_npz_fname= None,
                                            non_postprocessed_fname= None, force_separate_z= None,
                                            interpolation_order_z= 0, verbose= True,isbrats=False)         
            
            if opt.Conn_comp:
                output_Conn_Comp = join('./Output/',opt.dataroot,opt.encoder,opt.name,'Connect_components')
                maybe_mkdir_p(output_Conn_Comp)
                apply_postprocessing_to_folder(output_folder,output_Conn_Comp,for_which_classes=None)
            

            if step==datalen:
                break
            else:
                step+=1

        del val_outputs,val_outputs_seg
        
        labels= [i for i in range(opt.output_nc+1)] #improve to give name to the labels
        UseEvaluator=UseEvaluator_function(original_dir=join("./nnUNet/data/nnUnet_raw/nnUNet_raw_data",opt.dataroot,"labelsTr"), 
                                          files_to_copy= patientsID,labels=labels,predicted_dir=output_folder)
        
    
    #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝
    if  opt.enable_wandb:
        for i in labels[1:]:
            metrics=dict(dict(UseEvaluator['mean'])[str(i)])
            if opt.labels_name:
                object_seg=opt.labels_name[i]
            else:
                object_seg=str(i)
            opt.wandb_logger.log({"test/"+object_seg+"/Dice":metrics['Dice'],
                                     "test/"+object_seg+"/ASSD":metrics['Avg. Symmetric Surface Distance'],
                                     "test/"+object_seg+"/HD95":metrics['Hausdorff Distance 95'],
                                     "test/"+object_seg+"/VS":metrics['Volumetric Similarity'],
                                     "test/"+object_seg+"/Precision":metrics['Precision'],
                                     "test/"+object_seg+"/Recall":metrics['Recall'],
                                     "test/"+object_seg+"/Jaccard":metrics['Jaccard'],
                                    })
    #🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝🐝




