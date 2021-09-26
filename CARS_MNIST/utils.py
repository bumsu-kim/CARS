#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:51:51 2021

@author: (anonymous)
"""

# Some useful functions to keep tabs on the training, courtesy of Samy Wu Fung

import torch
from prettytable import PrettyTable
import torch.nn.functional as F

def get_stats(net, test_loader):
        test_loss=0
        correct=0
        with torch.no_grad():
            for d_test, labels in test_loader:
                batch_size = d_test.shape[0]
                y = net(d_test)  # apply the network to the test data
                test_loss += batch_size*F.nll_loss(y, labels).item() # sum up batch loss
                
                pred = y.argmax(dim=1,keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                
        test_loss /= len(test_loader.dataset)
        test_acc = 100.*correct/len(test_loader.dataset)
        
        return test_loss, test_acc, correct
                
        
def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['Total', num_params])
    return table