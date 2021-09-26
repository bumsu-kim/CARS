#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:49:15 2021

@author: (anonymous)
"""

# Defines the network architecture
import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5 , stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                              kernel_size=5, stride=1) 
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                              kernel_size=4, stride=1)
        self.fc_1 = nn.Linear(in_features=120,out_features=84)  
        self.fc_2 = nn.Linear(in_features=84, out_features=10)
            
    def forward(self,u):
        u = self.conv1(u)  # apply first convolutional layer
        u = self.relu(u)   # apply ReLU activation
        u = self.pool(u)   # apply max-pooling
        u = self.conv2(u)  # apply second convolutional layer
        u = self.relu(u)   # Apply ReLU activation
        u = self.pool(u)
        u = self.conv3(u)  # Apply third and final convolutional layer
        u = torch.flatten(u, 1)
        u = self.fc_1(u)
        u = self.relu(u)
        u = self.fc_2(u)
        u = self.relu(u)
        y = F.log_softmax(u, dim=1)
        return y
    