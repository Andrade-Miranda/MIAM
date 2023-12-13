import os
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
centre = "/home/guillaume/data4To/crossMoDA_2k22/2_preprocessedv2/manuscrit/source_preprocessed_A"
nb=105

imgfiles = os.listdir(centre)[:nb]
imgfiles.sort(key=natural_keys)
imgfiles = [centre+"/"+name for name in imgfiles]

nb_bins = 129
count = np.zeros((nb_bins,nb))

for (i,imgname) in enumerate(imgfiles):
    img = ((nib.load(imgname)).get_fdata()+1)*256/2 #put in range [0,256], it was strange with [-1,1]
    hist = np.histogram(img, bins=nb_bins, range=[0, 256])
    count[:,i] += hist[0]

mean = np.mean(count, axis=1)
std = np.std(count, axis=1)
x=np.linspace(0,256,nb_bins)
ymin=np.array([mean[i]-std[i] for i in range(mean.shape[0])])
ymax=np.array([mean[i]+std[i] for i in range(mean.shape[0])])
print(mean.shape)
print(std.shape)
print(x.shape)

bins = hist[1]
fig = plt.figure(figsize=(20,10))
plt.bar(bins[:-1], mean, color='g')
plt.plot(x,ymin, color='r')
plt.plot(x,ymax, color='r')
