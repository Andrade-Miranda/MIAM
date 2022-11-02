#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:44:13 2021

@author: gustavo
"""

import torch.nn as nn
import torch

import functional as F
import base
import statistics as st

def metricStatistics(Dice,ASD,HD,Recall,Precision):
    Dice_mu=st.mean(Dice)
    Dice_stdev=st.stdev(Dice)
    
    ASD_mu=st.mean(ASD)
    ASD_stdev=st.stdev(ASD)
    
    HD_mu=st.mean(HD)
    HD_stdev=st.stdev(HD)
    
    Recall_mu=st.mean(Recall)
    Recall_stdev=st.stdev(Recall)
    
    Precision_mu=st.mean(Precision)
    Precision_stdev=st.stdev(Precision)
    
    return (Dice_mu,Dice_stdev),(ASD_mu,ASD_stdev),(HD_mu,HD_stdev),(Recall_mu,Recall_stdev),(Precision_mu,Precision_stdev)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)




# class IoU(base.Metric):
#     __name__ = 'iou_score'

#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.iou(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )


# class Fscore(base.Metric):

#     def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.beta = beta
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.f_score(
#             y_pr, y_gt,
#             eps=self.eps,
#             beta=self.beta,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )


# class Accuracy(base.Metric):

#     def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.accuracy(
#             y_pr, y_gt,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )


# class Recall(base.Metric):

#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.recall(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )


# class Precision(base.Metric):

#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels

#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.precision(
#             y_pr, y_gt,
#             eps=self.eps,
#             threshold=self.threshold,
#             ignore_channels=self.ignore_channels,
#         )