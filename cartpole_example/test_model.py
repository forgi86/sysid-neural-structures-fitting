#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:25:22 2019

@author: marco
"""

import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(".."))
from torchid.neuralode import  NeuralODE, RunningAverageMeter
from torchid.ssmodels import MechanicalStateSpaceModel



# In[Load data]
if __name__ == '__main__':
    
    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = t[1] - t[0]
    x_noise = x
 
# In[Model]
    ss_model = MechanicalStateSpaceModel(Ts)
    nn_solution = NeuralODE(ss_model)
    model_name = "model_ARX_FE_sat.pkl"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_name)))    