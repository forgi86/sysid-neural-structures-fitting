import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from symbolic_RLC import fxu_ODE, fxu_ODE_mod
from neuralode import  NeuralODE, RunningAverageMeter
from ssmodels import NeuralStateSpaceModelLin
from symbolic_RLC import A_nominal, B_nominal


if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv("data/RLC_data_sat_FE.csv")

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])

    Ts = time_data[1] - time_data[0]
    t_fit = 2e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 20000
    test_freq = 10

    std_noise_V = 1.0*10
    std_noise_I = 1.0*1
    std_noise = np.array([std_noise_V, std_noise_I])

    input_data = u[0:n_fit]
    state_data = x[0:n_fit]
    output_data = y[0:n_fit]
    u_torch = torch.from_numpy(input_data)
    
    y_torch = torch.tensor(output_data)
    
    state_meas = np.copy(state_data)
    state_meas += np.random.randn(*state_meas.shape) * std_noise
    state_hidden_init = np.copy(state_meas) + np.random.randn(*state_meas.shape) * [1e-4, 1e-4]
    state_hidden_init = state_hidden_init.astype(np.float32)
    
    x_hidden_torch = torch.tensor(state_hidden_init, requires_grad=True)
    x_meas_torch = torch.tensor(state_meas, requires_grad=False)
    

    ss_model = NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralODE(ss_model)
    #nn_solution.load_state_dict(torch.load('model_ARX_FE_sat.pkl'))

    params = list(nn_solution.ss_model.parameters())
    params += [x_hidden_torch]
    optimizer = optim.Adam(params, lr=1e-4)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)


    scale_error = torch.tensor((std_noise).astype(np.float32))
    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()
        #x_pred_torch = nn_solution.f_ARX(x_hidden_torch, u_torch)
        #loss = torch.mean(torch.abs(x_pred_torch - x_true_torch))
        loss = 10000*nn_solution.f_ARX_consistency_loss(x_hidden_torch, u_torch) # consistency equation
        err_fit = x_meas_torch-x_hidden_torch
        err_fit_scaled = err_fit/scale_error
        loss += torch.mean((err_fit_scaled) **2) # fit equation
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % test_freq == 0:
            with torch.no_grad():
                x_pred_torch = nn_solution.f_onestep(x_hidden_torch, u_torch) #func(x_true_torch, u_torch)
                #loss = torch.mean(torch.abs(x_pred_torch - x_hidden_torch))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
        end = time.time()

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", "model_ARX_FE_sat.pkl"))

    x_0 = state_data[0,:]

    with torch.no_grad():
        x_sim = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(input_data))


    # In[Plot]
    x_hidden_torch = x_hidden_torch.detach()
    x_true_torch = torch.tensor(state_data)
    x_sim = np.array(x_sim)
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(np.array(x_true_torch[:,0]), 'k',  label='True')
    ax[0].plot(np.array(x_meas_torch[:,0]), 'k+',  label='Meas')
    ax[0].plot(np.array(x_hidden_torch[:,0]), 'g', label='Hidden')
    ax[0].plot(x_sim[:,0],'r', label='Sim')
    ax[0].legend()
    ax[1].plot(np.array(x_true_torch[:,1]), 'k',  label='True')
    ax[1].plot(np.array(x_meas_torch[:,1]), 'k+',  label='Meas')
    ax[1].plot(np.array(x_hidden_torch[:,1]), 'g', label='Hidden')
    ax[1].plot(x_sim[:,1],'r', label='Sim')
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
