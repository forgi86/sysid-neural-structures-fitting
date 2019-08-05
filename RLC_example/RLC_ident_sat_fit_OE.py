import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from symbolic_RLC import fxu_ODE, fxu_ODE_mod, A_nominal, B_nominal
from neuralode import  NeuralODE, RunningAverageMeter
from ssmodels import NeuralStateSpaceModelLin, NeuralStateSpaceModel

if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv(os.path.join("data", "RLC_data_FE.csv"))

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])

    Ts = time_data[1] - time_data[0]
    t_fit = 1e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 100
    test_freq = 10

    input_data = u[0:n_fit]
    state_data = x[0:n_fit]
    output_data = y[0:n_fit]
    u_torch = torch.from_numpy(input_data)
    x_true_torch = torch.from_numpy(state_data)

    ss_model = NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralODE(ss_model)

    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=1e-5)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()
        x_pred_torch = nn_solution.f_OE(x0_torch, u_torch)
        #y_pred_torch = torch.tensordot(x_pred_torch, C_matrix, ((-1,), (1,))) # we measure the first state
        loss = torch.mean((x_pred_torch - x_true_torch)**2)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % test_freq == 0:
            with torch.no_grad():
                x_pred_torch = nn_solution.f_OE(x0_torch, u_torch)
                loss = torch.mean(torch.abs(x_pred_torch - x_true_torch))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        end = time.time()

#    torch.save(nn_solution.state_dict(), 'model.pkl')

    t_val = 5e-3
    n_val = int(t_val//Ts)#x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]
    output_data_val = y[0:n_val]

    x0_val = np.zeros(2,dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)

    with torch.no_grad():
        x_pred_torch_val = nn_solution.f_OE(x0_torch_val, u_torch_val)

    # In[1]

    fig,ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(np.array(x_true_torch_val[:,0]), label='True')
    ax[0].plot(np.array(x_pred_torch_val[:,0]), label='Fit')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_true_torch_val[:,1]), label='True')
    ax[1].plot(np.array(x_pred_torch_val[:,1]), label='Fit')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_val), label='Input')
    ax[2].grid(True)
