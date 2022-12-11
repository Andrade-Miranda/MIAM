#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:23:08 2021

@author: gustavo
"""




import numpy as np
import matplotlib.pyplot as plt




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










