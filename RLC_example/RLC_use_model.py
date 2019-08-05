import os
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from symbolic_RLC import fxu_ODE, fxu_ODE_mod, A_nominal, B_nominal
from neuralode import  NeuralODE, RunningAverageMeter
from ssmodels import NeuralStateSpaceModelLin, NeuralStateSpaceModel


if __name__ == '__main__':

    Ts = 2e-7
    A_lin = A_nominal * Ts
    B_lin = B_nominal * Ts

    A_lin = np.array([[10, 100], [1000, 10000]])
    ss_model = NeuralStateSpaceModelLin(A_lin, B_lin)
    #ss_model = NeuralStateSpaceModel() #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralODE(ss_model)
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model.pkl")))

    # In[Linearization time - 1 a time]
    nx = 2
    nu = 1

    x_arr = np.random.rand(nx).astype(np.float32)
    x_torch = torch.tensor(x_arr, requires_grad=True)

    u_batch = np.random.rand(nu).astype(np.float32)
    u_torch = torch.tensor(u_batch, requires_grad=True)


    VAR = []
    for idx_var in range(nx):
        var = np.zeros((1,nx)).astype(np.float32)
        var[0,idx_var] = 1.0 # differentiate w.r.t the nth variable
        VAR.append(torch.tensor(var))

    F_xu = ss_model(x_torch,u_torch)
    A = np.empty((nx,nx))
    B = np.empty((nx,nu))
        
    for idx_var in range(nx):
        var = VAR[idx_var]
        #var = np.zeros((1,nx)).astype(np.float32)
        #var[0,idx_var] = 1.0 # differentiate w.r.t the nth variable
        F_xu.backward(var, retain_graph=True)
        A[idx_var,:] = np.array(x_torch.grad)
        B[idx_var,:] = np.array(u_torch.grad)
        x_torch.grad.data.zero_()
        u_torch.grad.data.zero_()


    # In[Linearization time - batched]


    batch_size = 64
    x_batch = np.random.rand(batch_size, nx).astype(np.float32)
    x_torch = torch.tensor(x_batch, requires_grad=True)

    u_batch = np.random.rand(batch_size, nu).astype(np.float32)
    u_torch = torch.tensor(u_batch, requires_grad=True)

    
    
    VAR = []
    for idx_var in range(nx):
        var = np.zeros((batch_size,nx)).astype(np.float32)
        var[:,idx_var] = 1.0 # differentiate w.r.t the nth variable
        VAR.append(torch.tensor(var))

    # In[Linearization time - batched]
    

    F_xu = ss_model(x_torch,u_torch)
    A = np.empty((batch_size,nx,nx))
    B = np.empty((batch_size,nx,nu))
    for idx_var in range(nx):
        var = VAR[idx_var]
        F_xu.backward(var, retain_graph=True)
        A[:,idx_var,:] = np.array(x_torch.grad)
        B[:,idx_var,:] = np.array(u_torch.grad)
        x_torch.grad.data.zero_()
        u_torch.grad.data.zero_()
