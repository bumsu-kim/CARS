#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:47:36 2021

@author: (anonymous)
"""

# Defines the data loaders

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch 

def MNIST_Loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    
    normalize = transforms.Normalize((0.,), (1.,)) # unnormalized?
    #normalize = transforms.Normalize((0.1307,), (0.3081,))
    Clean = transforms.Compose([transforms.ToTensor(), normalize])
   
    #!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    #!tar -zxvf MNIST.tar.gz
    
    train_data = datasets.MNIST('./', train=True,
                                   download=True, transform=Clean)
    test_data = datasets.MNIST('./', train=False,
                                  download=True, transform=Clean)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                    batch_size=train_batch_size)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                    batch_size=test_batch_size)
    
    return train_loader, test_loader