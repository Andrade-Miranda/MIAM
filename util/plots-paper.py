#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:53:59 2022

@author: gustavoandrade
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

####BRATS2021#####
###DICE AVERAGE####################
xdata = np.array([10,20, 30, 40, 50, 60, 70, 80, 90, 100])

##CNN_{h}+ViT_{v}-{B}/1
yCNN = np.array([0.779, 0.818, 0.832,	0.851,	0.858,	0.856,	0.867,	0.868,	0.871,	0.886])*100
dyCNN = np.array([0.005,0.013,	0.012,	0.007,	0.008,	0.014,	0.012,	0.006,	0.012,	0.002])*100

yMCNN = np.array([0.787,0.825, 0.843, 0.855,  0.864,  0.866,  0.879,  0.883	,0.884	,0.885])*100
dyMCNN = np.array([0.005,0.008,	0.009,	0.007,	0.002,	0.012,	0.004,	0.003,	0.005,	0.003])*100

UNETR = np.array([0.719,	0.765,	0.785,	0.786, 	0.804,	0.801,	0.815,	0.825,	0.819,	0.824])*100
dyUNETR = np.array([0.045,	0.010,	0.015,	0.006,	0.006,	0.012,	0.007,	0.021,	0.022,	0.018])*100

swin = np.array([0.793,	0.821,	0.856,	0.866,	0.871,	0.878,	0.883,	0.883,	0.882,	0.890])*100
dyswin = np.array([0.002,	0.006,	0.008,	0.007,	0.004,	0.003,	0.000,	0.003,	0.000,	0.000])*100


# Visualize the result
plt.plot(xdata, yCNN, '-or', color='blue')
plt.fill_between(xdata, yCNN - dyCNN, yCNN + dyCNN,
                 color='blue', alpha=0.2)

plt.plot(xdata, yMCNN, '-or', color='red')
plt.fill_between(xdata, yMCNN - dyMCNN, yMCNN + dyMCNN,
                 color='red', alpha=0.2)

plt.plot(xdata, UNETR, '-or', color='green')
plt.fill_between(xdata, UNETR - dyUNETR, UNETR + dyUNETR,
                 color='green', alpha=0.2)

plt.plot(xdata, swin, '-or', color='orange')
plt.fill_between(xdata, swin - dyswin, swin + dyswin,
                 color='orange', alpha=0.2)


plt.xlabel(r'$\textrm{\# samples}$',fontdict=font)
plt.ylabel(r'$\textrm{Avg. Dice}$',fontdict=font)
plt.xlim(0, 110);
plt.ylim(0.71*100, 0.90*100);
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend([r"$\textrm{CNN+VIT}_{v}\text{-{B}/1}$",r"$\textrm{MCNN+VIT}_{v}\text{-{B}/1}$",r'$\textrm{UNETR}$',r'$\textrm{SwinUNETR}$'])
plt.savefig('BratsAvgDICE.pdf')

########################################################################################################################
###DICE ET####################
xdata = np.array([10,20, 30, 40, 50, 60, 70, 80, 90, 100])

##CNN_{h}+ViT_{v}-{B}/1
yCNN = np.array([0.752,	0.801,	0.807,	0.819,	0.825,	0.822,	0.832,	0.837,	0.846,	0.858])*100
dyCNN = np.array([0.007,	0.011,	0.006,	0.012,	0.007,	0.011,	0.016,	0.004,	0.016,	0.002])*100

yMCNN = np.array([0.772,	0.806,	0.813,	0.824,	0.837,	0.837,	0.845,	0.847,	0.854,	0.854])*100
dyMCNN = np.array([0.002,	0.010,	0.011,	0.016,	0.004,	0.019,	0.008,	0.004,	0.003,	0.004])*100

UNETR = np.array([0.749,	0.760,	0.773,	0.771,	0.793,	0.784,	0.794,	0.804,	0.794,	0.802])*100
dyUNETR = np.array([0.007,	0.009,	0.013,	0.010,	0.010,	0.012,	0.005,	0.019,	0.021,	0.010])*100

swin = np.array([0.772,	0.795,	0.830,	0.837,	0.849,	0.848,	0.854,	0.855,	0.848,	0.857])*100
dyswin = np.array([0.003,	0.010,	0.007,	0.008,	0.002,	0.001,	0.004,	0.004,	0.000,	0.000])*100


# Visualize the result
plt.plot(xdata, yCNN, '-or', color='blue')
plt.fill_between(xdata, yCNN - dyCNN, yCNN + dyCNN,
                 color='blue', alpha=0.2)

plt.plot(xdata, yMCNN, '-or', color='red')
plt.fill_between(xdata, yMCNN - dyMCNN, yMCNN + dyMCNN,
                 color='red', alpha=0.2)

plt.plot(xdata, UNETR, '-or', color='green')
plt.fill_between(xdata, UNETR - dyUNETR, UNETR + dyUNETR,
                 color='green', alpha=0.2)

plt.plot(xdata, swin, '-or', color='orange')
plt.fill_between(xdata, swin - dyswin, swin + dyswin,
                 color='orange', alpha=0.2)


plt.xlabel(r'$\textrm{\# samples}$',fontdict=font)
plt.ylabel(r'$\textrm{ET Dice}$',fontdict=font)
plt.xlim(0, 110);
plt.ylim(0.74*100, 0.87*100);
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend([r"$\textrm{CNN+VIT}_{v}\text{-{B}/1}$",r"$\textrm{MCNN+VIT}_{v}\text{-{B}/1}$",r'$\textrm{UNETR}$',r'$\textrm{SwinUNETR}$'])
plt.savefig('BratsDICEET.pdf')
########################################################################################################################

########################################################################################################################
###DICE WT####################
xdata = np.array([10,20, 30, 40, 50, 60, 70, 80, 90, 100])

##CNN_{h}+ViT_{v}-{B}/1
yCNN = np.array([0.837,	0.856,	0.872,	0.887,	0.888,	0.885,	0.897,	0.888,	0.884,	0.902])*100
dyCNN = np.array([0.007,	0.011,	0.015,	0.003,	0.004,	0.010,	0.009,	0.010,	0.005,	0.006])*100

yMCNN = np.array([0.832,	0.846,	0.865,	0.870,	0.886,	0.881,	0.903,	0.905,	0.903,	0.905])*100
dyMCNN = np.array([0.003,	0.010,	0.003,	0.008,	0.005,	0.006,	0.004,	0.006,	0.008,	0.004])*100

UNETR = np.array([0.810,	0.821,	0.839,	0.829,	0.854,	0.838,	0.853,	0.862,	0.854,	0.859])*100
dyUNETR = np.array([0.004,	0.006,	0.006,	0.003,	0.015,	0.009,	0.008,	0.025,	0.016,	0.017])*100

swin = np.array([0.851,	0.848,	0.878,	0.887,	0.885,	0.897,	0.894,	0.894,	0.900,	0.905])*100
dyswin = np.array([0.003,	0.004,	0.013,	0.010,	0.010,	0.003,	0.007,	0.007,	0.000,	0.000])*100


# Visualize the result
plt.plot(xdata, yCNN, '-or', color='blue')
plt.fill_between(xdata, yCNN - dyCNN, yCNN + dyCNN,
                 color='blue', alpha=0.2)

plt.plot(xdata, yMCNN, '-or', color='red')
plt.fill_between(xdata, yMCNN - dyMCNN, yMCNN + dyMCNN,
                 color='red', alpha=0.2)

plt.plot(xdata, UNETR, '-or', color='green')
plt.fill_between(xdata, UNETR - dyUNETR, UNETR + dyUNETR,
                 color='green', alpha=0.2)

plt.plot(xdata, swin, '-or', color='orange')
plt.fill_between(xdata, swin - dyswin, swin + dyswin,
                 color='orange', alpha=0.2)


plt.xlabel(r'$\textrm{\# samples}$',fontdict=font)
plt.ylabel(r'$\textrm{WT Dice}$',fontdict=font)
plt.xlim(0, 110);
plt.ylim(0.80*100, 0.92*100);
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend([r"$\textrm{CNN+VIT}_{v}\text{-{B}/1}$",r"$\textrm{MCNN+VIT}_{v}\text{-{B}/1}$",r'$\textrm{UNETR}$',r'$\textrm{SwinUNETR}$'])
plt.savefig('BratsDICEWT.pdf')
########################################################################################################################


########################################################################################################################
###DICE TC####################
xdata = np.array([10,20, 30, 40, 50, 60, 70, 80, 90, 100])

##CNN_{h}+ViT_{v}-{B}/1
yCNN = np.array([0.747,	0.797,	0.816,	0.847,	0.860,	0.862,	0.873,	0.880,	0.882,	0.898])*100
dyCNN = np.array([0.008,	0.033,	0.016,	0.013,	0.013,	0.023,	0.018,	0.010,	0.017,	0.002])*100

yMCNN = np.array([0.756,	0.823,	0.852,	0.872,	0.870,	0.880,	0.890,	0.896,	0.896,	0.897])*100
dyMCNN = np.array([0.017,	0.014,	0.015,	0.004,	0.004,	0.010,	0.004,	0.002,	0.004,	0.003])*100

UNETR = np.array([0.597,	0.714,	0.744,	0.746,	0.783,	0.781,	0.798,	0.808,	0.808,	0.810])*100
dyUNETR = np.array([0.142,	0.022,	0.032,	0.014,	0.008,	0.017,	0.012,	0.020,	0.032,	0.027])*100

swin = np.array([0.757,	0.816,	0.862,	0.875,	0.882,	0.892,	0.900,	0.903,	0.899,	0.909])*100
dyswin = np.array([0.014,	0.005,	0.007,	0.005,	0.004,	0.007,	0.003,	0.005,	0.000,	0.000])*100


# Visualize the result
plt.plot(xdata, yCNN, '-or', color='blue')
plt.fill_between(xdata, yCNN - dyCNN, yCNN + dyCNN,
                 color='blue', alpha=0.2)

plt.plot(xdata, yMCNN, '-or', color='red')
plt.fill_between(xdata, yMCNN - dyMCNN, yMCNN + dyMCNN,
                 color='red', alpha=0.2)

plt.plot(xdata, UNETR, '-or', color='green')
plt.fill_between(xdata, UNETR - dyUNETR, UNETR + dyUNETR,
                 color='green', alpha=0.2)

plt.plot(xdata, swin, '-or', color='orange')
plt.fill_between(xdata, swin - dyswin, swin + dyswin,
                 color='orange', alpha=0.2)


plt.xlabel(r'$\textrm{\# samples}$',fontdict=font)
plt.ylabel(r'$\textrm{TC Dice}$',fontdict=font)
plt.xlim(0, 110);
plt.ylim(0.59*100, 0.92*100);
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend([r"$\textrm{CNN+VIT}_{v}\text{-{B}/1}$",r"$\textrm{MCNN+VIT}_{v}\text{-{B}/1}$",r'$\textrm{UNETR}$',r'$\textrm{SwinUNETR}$'])
plt.savefig('BratsDICETC.pdf')
########################################################################################################################



####HEKTOR2021#####
###DICE AVERAGE####################
xdata = np.array([10,20, 30, 40, 50, 60, 70, 80, 90, 100])

##CNN_{h}+ViT_{v}-{B}/1
yCNN = np.array([62.7,	66.7,	68.0,	69.5,	70.5,	70.9,	71.5,	72.0,	72.1,	72.0])
dyCNN = np.array([2.9,	2.8,	2.1,	1.9,	1.3,	0.8,	0.6,	1.0,	0.8,	0.5])

yMCNN = np.array([64.0,	67.6,	68.5,	69.8,	70.8,	71.0,	71.5,	71.6,	72.0,	72.3])
dyMCNN = np.array([2.5,	2.0,	1.8,	1.1,	0.7, 	0.5,	1.2,	1.0,	0.9,	0.8])

UNETR = np.array([62.4,	65.6,	66.7,	68.2,	68.8,	69.7,	69.9,	70.6,	70.7,	70.8])
dyUNETR = np.array([2.3,	1.7,	1.3,	1.9,	1.9,	1.7,	1.2,	1.1,	1.9,	1.2])

swin = np.array([66.2,	68.3,	69.8,	70.3,	70.9,	71.4,	72.0,	72.2,	72.5,	72.9])
dyswin = np.array([2.6, 	0.8,	0.8,	0.9,	1.8,	1.3,	0.5,	1.0,	0.9,	0.7])


# Visualize the result
plt.plot(xdata, yCNN, '-or', color='blue')
plt.fill_between(xdata, yCNN - dyCNN, yCNN + dyCNN,
                 color='blue', alpha=0.2)

plt.plot(xdata, yMCNN, '-or', color='red')
plt.fill_between(xdata, yMCNN - dyMCNN, yMCNN + dyMCNN,
                 color='red', alpha=0.2)

plt.plot(xdata, UNETR, '-or', color='green')
plt.fill_between(xdata, UNETR - dyUNETR, UNETR + dyUNETR,
                 color='green', alpha=0.2)

plt.plot(xdata, swin, '-or', color='orange')
plt.fill_between(xdata, swin - dyswin, swin + dyswin,
                 color='orange', alpha=0.2)


plt.xlabel(r'$\textrm{\# samples}$',fontdict=font)
plt.ylabel(r'$\textrm{Avg. Dice}$',fontdict=font)
plt.xlim(0, 110);
plt.ylim(62, 73);
plt.xticks([10,20,30,40,50,60,70,80,90,100])
plt.legend([r"$\textrm{CNN+VIT}_{v}\text{-{B}/1}$",r"$\textrm{MCNN+VIT}_{v}\text{-{B}/1}$",r'$\textrm{UNETR}$',r'$\textrm{SwinUNETR}$'])
plt.savefig('HEKTORAvgDICE.pdf')











