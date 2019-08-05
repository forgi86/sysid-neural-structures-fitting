# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:00:29 2019

@author: marco
"""
import torch
import torch.nn as nn
import numpy as np



class NeuralStateSpaceModel(nn.Module):
    def __init__(self):
        super(NeuralStateSpaceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(64,2)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, X,U):
        XU = torch.cat((X,U),-1)
        DX = self.net(XU)
        return DX
    
    
class NeuralStateSpaceModelLin(nn.Module):
    def __init__(self, AL, BL):
        super(NeuralStateSpaceModelLin, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(64,2)
        )

        self.AL = nn.Linear(2,2, bias=False)
        self.AL.weight = torch.nn.Parameter(torch.tensor(AL.astype(np.float32)), requires_grad=False)
        self.BL = nn.Linear(1,2, bias=False)
        self.BL.weight = torch.nn.Parameter(torch.tensor(BL.astype(np.float32)), requires_grad=False)


        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, X,U):
        XU = torch.cat((X,U),-1)
        DX = self.net(XU)
        DX += self.AL(X) + self.BL(U)
        return DX   
