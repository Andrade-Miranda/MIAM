#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:23:08 2021

@author: gustavo
"""




import numpy as np
import matplotlib.pyplot as plt

import wandb

#/*
 ############ECE Plot#############################
ax1 = plt.subplot(2, 1, 1)
prob_true, prob_pred = calibration_curve(y_test,y_prob, n_bins=10,strategy='uniform')
ax1.grid()
ax1.set_title("ECE Calibration plot")
self.dispCER_ECE = CalibrationDisplay(prob_true, prob_pred,y_prob)
ax1.plot(prob_pred, prob_true,marker='o',label='Calibration plots')
ax1.text(0.01, 0.95, 'ECE= '+ECE, fontsize=8, ha='left', va='top', color='blue')
ax1.plot([0,1],[0,1],linestyle='--', color='gray',label='Perfect calibration')
# Add histogram
ax2 = plt.subplot(2, 1, 2)
ax2.hist(self.dispCER_ECE.y_prob,
            range=(0, 1),
            bins=15,
            label="",#self.opt.name,
            )
ax2.set(title="Histogram calibration plot", xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(self.opt.outputSoft_dir,'ECE.pdf'))
plt.close()
self.plot_calibration_curve(y_test, y_prob,strategy='uniform',title='ECE Plot',ECE_value=ECE, n_bins=20, ax=None, hist=True, normalize=False)
############ADA ECE Plot#############################
ax1 = plt.subplot(2, 1, 1)
prob_trueADA, prob_predADA = calibration_curve(y_test,y_prob, n_bins=15,strategy='quantile')
ax1.grid()
ax1.set_title("ADA ECE Calibration plot")
self.dispCER_ADAECE = CalibrationDisplay(prob_trueADA, prob_predADA,y_prob)
ax1.plot(prob_predADA, prob_trueADA,marker='o',label='Calibration plots')
ax1.text(0.05, 0.95, 'ECE= '+ADA_ECE, fontsize=8, ha='left', va='top', color='blue')
ax1.plot([0,1],[0,1],linestyle='--', color='gray',label='Perfect calibration')
# Add histogram
ax2 = plt.subplot(2, 1, 2)
ax2.hist(self.dispCER_ADAECE.y_prob,
            range=(0, 1),
            bins=15,
            label="",#self.opt.name,
            )
ax2.set(title="Histogram calibration plot", xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(self.opt.outputSoft_dir,'ADA_ECE.pdf'))
plt.close()
#*/
################plot wandb tables 

    # 🐝 create a wandb table to log input image, ground_truth masks and predictions
    #columns = ["filename", "image", "ground_truth", "prediction"]
    #tableSeg = wandb.Table(columns=columns)
    
    # # 🐝
    # if  opt.enable_wandb:
    #     # Create a table with the columns to plot
    #     all_valid_keys=[metricspercase[i]['filename'].split('/')[-1].split('.')[0] for i in range(len(metricspercase))]
    #     val_dice=[metricspercase[i]['dice'][0] for i in range(len(metricspercase))]
    #     id=list(range(len(metricspercase)))
    #     data = [[case,x, y] for (case,x,y) in zip(all_valid_keys,id,val_dice)]
    #     table = wandb.Table(data=data, columns = ["CasesID","ID","DSC"])
    #     wandb.log({"ValScatter/plot" : wandb.plot.scatter(table, "ID", "DSC",
    #                              title="Val cases vs DSC Scatter Plot")})
        
    #     data = [[case,y] for (case,y) in zip(all_valid_keys,val_dice)]
    #     table = wandb.Table(data=data, columns = ["CasesID","DSC"])
    #     wandb.log({"ValBar/plot" : wandb.plot.bar(table, "CasesID", "DSC",
    #                              title="Val cases vs DSC bar"),
    #                  "Test prediction" :tableSeg})



    # # 🐝
    # if  opt.enable_wandb:
    #     opt.wandb_logger.log({
    #             "rocTest" : wandb.plot.roc_curve([clasif_metrics.case_target[s] for s in clasif_metrics.subject_list],
    #                                                     [[1-clasif_metrics.case_pred[s],clasif_metrics.case_pred[s]] for s in clasif_metrics.subject_list],
    #                                                     title='ROC Test',classes_to_plot=1),
    #             "prTest":wandb.plot.pr_curve([clasif_metrics.case_target[s] for s in clasif_metrics.subject_list], 
    #                                                [[1-clasif_metrics.case_pred[s],clasif_metrics.case_pred[s]] for s in clasif_metrics.subject_list],
    #                                                title='Precision vs Recall Test',classes_to_plot=1),
    #             "Confusion Matrix Test WB":wandb.plot.confusion_matrix(
    #                                      y_true=[clasif_metrics.case_target[s] for s in clasif_metrics.subject_list],
    #                                     preds=[np.argmax([1-clasif_metrics.case_pred[s],clasif_metrics.case_pred[s]]) for s in clasif_metrics.subject_list],     
    #                                     class_names=['Benign','Malign']),
    #             "Confusion Matrix": wandb.sklearn.plot_confusion_matrix(y_true=[clasif_metrics.case_target[s] for s in clasif_metrics.subject_list],
    #                                                                              y_pred=[np.argmax([1-clasif_metrics.case_pred[s],clasif_metrics.case_pred[s]]) for s in clasif_metrics.subject_list], 
    #                                                                              labels=['Benign','Malign'])
    #                                     }) 





##########Plots matplots in wandb#############################

        #### FROC plots
        # data = [[x, y] for (x, y) in zip(self.fp_per_case, self.sensitivity)]
        # table = wandb.Table(data=data, columns=["False positives per case", "Sensitivity"])
        # self.wandb_logger.log(
        #     {
        #     "FROC": wandb.plot.line(table, "False positives per case", "Sensitivity", title="FROC")
        #     }
        # )           
        # #### calibration plots
        # random_guessing = np.linspace(0, 1, len(self.dispCER.prob_pred))
        # # Combine Calibration diagram and Random Guessing in one plot
        # self.wandb_logger.log({"Calibration diagram": wandb.plot.line_series(
        #                         xs=[list(self.dispCER.prob_pred)] + [list(random_guessing)],
        #                         ys=[list(self.dispCER.prob_true)] + [list(random_guessing)],
        #                         keys=['Calibration','Reference'],
        #                         title="Calibration diagram",
        #                         xname='Mean confidence',
        #                         )})
        # y_prob=np.array([[metrics.case_pred[s]] for s in metrics.subject_list])
        # y_test=np.array([metrics.case_target[s] for s in metrics.subject_list])
        # # Combine Calibration diagram using plt
        # f, ax = plt.subplots()
        # prob_true, prob_pred = calibration_curve(y_test,y_prob, n_bins=10)
        # ax.grid()
        # ax.set_title("Calibration diagram-plt")
        # self.dispCER = CalibrationDisplay(prob_true, prob_pred,y_prob)
        # ax.plot(prob_pred, prob_true,marker='o',label='Calibration plots')
        # ax.plot([0,1],[0,1],linestyle='--', color='gray',label='Perfect calibration')
        # plt.tight_layout()
        # plt.show()
        # self.wandb_logger.log({"Calibration diagram-plt": plt})

        # # Add histogram
        # f, ax = plt.subplots()
        # ax.hist(self.dispCER.y_prob,
        #             range=(0, 1),
        #             bins=20,
        #             label=self.opt.name,
        #             )
        # ax.set(title=self.opt.name, xlabel="Mean predicted probability", ylabel="Count")
        # plt.tight_layout()
        # plt.show()
        # self.wandb_logger.log({"Calibration Histogram-plt": plt})

















fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        img=inputs[0,i,:,:,50].detach().numpy()
        plt.imshow(img,cmap='gray')
    elif i in range(4,7):
        img=labels[0,i-4,:,:,50].detach().numpy()
        plt.imshow(img,cmap='gray')
    else:
        break
    
plt.show()



fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 3
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        img=val_inputs[0,i,:,:,60].detach().numpy()
        plt.imshow(img,cmap='gray')
    elif i in range(4,7):
        img=val_data['label'][0,i-4,:,:,60].detach().numpy()
        plt.imshow(img,cmap='gray')
    elif i in range(7,10):
        img=val_outputs[i-7,:,:,60].detach().numpy()
        plt.imshow(img,cmap='gray')
    
plt.show()








fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(2):
        img=val_inputs[0,i,:,:,70].detach().numpy()
        plt.imshow(img,cmap='gray')
    elif i==2:
        img=val_labels[0,0,:,:,70].detach().numpy()
        plt.imshow(img,cmap='gray')
    else:
        break
    
plt.show()


fig=plt.figure(figsize=(8, 8))
plt.imshow(val_outputs[0][0,:,:,70],cmap='gray')


fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(2):
        a=img['data'][0,i,60,:,:]
        plt.imshow(a,cmap='gray')
    elif i in range(2,3):
        a=img['seg'][0,0,60,:,:]
        plt.imshow(a,cmap='gray')
    
plt.show()





fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        img=inputs[0,i,65,:,:].detach().numpy()
        plt.imshow(img,cmap='gray')
    elif i in range(4,8):
        img=labels[0,i-4,65,:,:].detach().numpy()
        plt.imshow(img,cmap='gray')
    
plt.show()


fig=plt.figure(figsize=(8, 8))
for i in range(5):
    fig.add_subplot(1, 5, i+1)
    img=b['data'][i,85,:,:]
    plt.imshow(img,cmap='gray')
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        img=a['data'][0,0,85,:,:]
        plt.imshow(img,cmap='gray')
    elif i in range(4,8):
        img=a['seg'][0,0,85,:,:]
        plt.imshow(img,cmap='gray')
    
plt.show()



fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img=val_outputs[0][i,85,:,:]
    img=np.moveaxis(img.detach().numpy(),[0,1],[-1,0])
    plt.imshow(img)
    
plt.show()

fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    img=val_labels[0][i,85,:,:]
    img=np.moveaxis(img.detach().numpy(),[0,1],[-1,0])
    plt.imshow(img)
    
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        img=a['data'][0,i,80,:,:]
        plt.imshow(img,cmap='gray')
    elif i in range(4,8):
        img=a['seg'][0,0,80,:,:]
        plt.imshow(img,cmap='gray')
    
plt.show()



####load .pkl
proper=dataset['BraTS2021_00000']['properties_file']
with open(proper, 'rb') as f:
    plansProper = pickle.load(f)

data=dataset['BraTS2021_00000']['data_file']
img=np.load(data)
d1 = img['data']
fig=plt.figure(figsize=(8, 8))
for i in range(d1.shape[0]):
    fig.add_subplot(1, d1.shape[0], i+1)
    plt.imshow(d1[i,60,:,:],cmap='gray')






#### load npz
import numpy as np
img=np.load(self.dataset_tr['prostate_01']['data_file'])
val=np.load(self.dataset_val['prostate_00']['data_file'])

img=np.load(self.dataset_tr['BraTS2021_00000']['data_file'])
img1 = img['data']
val=np.load(self.dataset_val['BraTS2021_00006']['data_file'])
val1 = val['data']

fig=plt.figure(figsize=(8, 8))
for i in range(img1.shape[0]):
    fig.add_subplot(1, img1.shape[0], i+1)
    plt.imshow(img1[i,60,:,:],cmap='gray')



a='/home/gustavo/Code/CNNTrans/nnUNet/data/nnUnet_preprocessed/Task005_Prostate/nnUNetData_plans_v2.1_stage0/prostate_01.npz' 
img1=np.load(a)
d1 = img1['data']
fig=plt.figure(figsize=(8, 8))
for i in range(d.shape[0]):
    fig.add_subplot(1, d.shape[0], i+1)
    plt.imshow(d[i,60,:,:],cmap='gray')




fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 3
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(data['B'][i,0,:,:],cmap='gray')
    elif i in range(4,8):
        plt.imshow(data['Seg'][i-4,0,:,:],cmap='gray',vmin=0, vmax=1)
    else:
        plt.imshow(data['A'][i-8,0,:,:],cmap='gray')
    
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(outputs[0,i,80,:].detach().numpy(),cmap='gray',vmin=0, vmax=1)

plt.show()








fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        img=input['A'][i,0,:,:]
        plt.imshow(img,cmap='gray')
    elif i in range(4,8):
        img=input['B'][i-4,0,:,:]
        plt.imshow(img,cmap='gray')
    
plt.show()



fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(visuals['real_A'][:,:,i],cmap='gray')
    else:
        plt.imshow(visuals['fake_B'][:,:,i-4],cmap='gray')
    
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(target[i,:,:],cmap='gray')

###################LOSSSSSS########################################################
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(input.detach().numpy()[i,0,:,:],cmap='gray')
    elif i in range(4,8):
        plt.imshow(target.detach().numpy()[i-4,0,:,:],cmap='gray',vmin=0, vmax=1)
    elif i in range(8,12):
        plt.imshow(input.detach().numpy()[i-8,1,:,:],cmap='gray')
    else:
        plt.imshow(target.detach().numpy()[i-12,1,:,:],cmap='gray',vmin=0, vmax=1)
    
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(probs.detach().numpy()[i,0,:,:],cmap='gray')
    elif i in range(4,8):
        plt.imshow(target.detach().numpy()[i-4,0,:,:],cmap='gray',vmin=0, vmax=1)    
plt.show()



fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(probs.detach().numpy()[i,0,:,:],cmap='gray')
    elif i in range(4,8):
        plt.imshow(target.detach().numpy()[i-4,0,:,:],cmap='gray',vmin=0, vmax=1)
    elif i in range(8,12):
        plt.imshow(probs.detach().numpy()[i-8,1,:,:],cmap='gray')
    else:
        plt.imshow(target.detach().numpy()[i-12,1,:,:],cmap='gray',vmin=0, vmax=1)
    
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 4
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(num.detach().numpy()[i,0,:,:],cmap='gray')
    elif i in range(4,8):
        plt.imshow(target.detach().numpy()[i-4,0,:,:],cmap='gray',vmin=0, vmax=1)
    elif i in range(8,12):
        plt.imshow(num.detach().numpy()[i-8,1,:,:],cmap='gray')
    else:
        plt.imshow(target.detach().numpy()[i-12,1,:,:],cmap='gray',vmin=0, vmax=1)
    
plt.show()


fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(self.seg_fake_B[0][i,0,:,:].detach().numpy(),cmap='gray')
    else:
        plt.imshow(self.seg_fake_B[0][i-4,1,:,:].detach().numpy(),cmap='gray')
    
plt.show()

fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    if i in range(4):
        plt.imshow(self.real_A[i,0,:,:].detach().numpy(),cmap='gray')
    
plt.show()



fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(torch.max(self.seg_fake_B[0].data,dim=1,keepdim=True)[1][i,0,:,:].detach().numpy())

    
plt.show()



fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 1
for i in range(columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(image_numpy[:,:,0,i])

    
plt.show()











for epoch in range(1):
    l = []
    for batch_id, batch in enumerate(train_loader):
        print(f"Batch {batch_id+1}")

        # Let's store the batches to analyze them later
        l.extend(batch)      

    # How many samples were returned by the augmentor?
    print(f"Total samples returned by augmentor: {len(l)}")

    # How much of the original data was covered in the batches return?
    #print(f"Coverage of samples returned by augmentor of original dataset: {len(set(l).intersection(set(range(len(data)))))}/{len(data)}")










