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

from RLC_example.symbolic_RLC import fxu_ODE, fxu_ODE_mod, A_nominal, B_nominal
from torchid.ssfitter import  NeuralODE
from torchid.util import RunningAverageMeter
from torchid.ssmodels import NeuralStateSpaceModelLin, NeuralStateSpaceModel


if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))
    #df_X = pd.read_csv("RLC_data.csv")
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])


    std_noise_V = 10.0
    std_noise_I = 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    
    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    
    
    Ts = t[1] - t[0]
    t_fit = 5e-3
    n_fit = int(t_fit/Ts)#x.shape[0]
    num_iter = 10000
    seq_len = 100 #int(n_fit/10)
    batch_size = n_fit//seq_len
#    batch_size = 128
    test_freq = 10

    # Get fit data #
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    y_true_torch_fit = torch.from_numpy(y_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    x_hidden_init = np.copy(x_fit)
    x_hidden_init = x_hidden_init.astype(np.float32)
    x_hidden_torch_fit = torch.tensor(x_hidden_init, requires_grad=True)


    def get_batch(batch_size, seq_len):
        num_train_samples = x_meas_torch_fit.shape[0]
        s = torch.from_numpy(np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False))
        batch_x0_hidden = x_hidden_torch_fit[s, :]  # (M, D)
        batch_t = torch.stack([time_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x_meas = torch.stack([x_meas_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x_hidden = torch.stack([x_hidden_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_u = torch.stack([u_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        
        return batch_t, batch_x0_hidden, batch_u, batch_x_meas, batch_x_hidden
    

    ss_model = NeuralStateSpaceModel() #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralODE(ss_model)
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model.pkl")))

    params = list(nn_solution.ss_model.parameters()) + [x_hidden_torch_fit]
    optimizer = optim.Adam(params, lr=1e-3)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)


#    scale_error = torch.tensor((std_noise).astype(np.float32))
#    scale_error = torch.tensor((np.std(x_fit, axis=0)).astype(np.float32))

    scale_error = 1./np.std(x_fit, axis=0)


    
    ii = 0
    for itr in range(0, num_iter):


        if  itr > 0 and itr % test_freq == 0:
            with torch.no_grad():
                #x_pred_torch = nn_solution.f_OE(x0_torch, u_torch_fit)
                #loss = torch.mean(torch.abs(x_pred_torch - x_meas_torch_fit))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        optimizer.zero_grad()
        
        batch_t, batch_x0_hidden, batch_u, batch_x_meas, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_pred = nn_solution.f_OE_minibatch(batch_x0_hidden, batch_u)
        err = batch_x_meas[:,0:,:] - batch_x_pred[:,0:,:]
        err_scaled = err * scale_error
        loss_pred = torch.mean((err_scaled)**2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))
        
        err_hidden = batch_x_hidden[:,1:,:] - batch_x_pred[:,1:,:]
        err_scaled_hidden = err_hidden * scale_error
        loss_hidden = torch.mean((err_scaled_hidden)**2)
        
        loss = 1/2*(loss_pred + loss_hidden)
        
        loss.backward()
        #params[-1].grad
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())


        end = time.time()

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", "model_minibatch_OE_hidden.pkl"))

    # In[Simulation performance]
    x0_fit = np.zeros(2,dtype=np.float32)
    x0_torch_fit = torch.from_numpy(x0_fit)
    with torch.no_grad():
        x_sim_torch_fit = nn_solution.f_OE(x0_torch_fit, u_torch_fit)

    x_true_fit = x[0:n_fit]
    # In[FIT]
    

    fig,ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(np.array(np.array(x_meas_torch_fit[:,0].detach())), 'k*', label='Measured')
    ax[0].plot(np.array(np.array(x_hidden_torch_fit[:,0].detach())), 'b', label='Hidden')
    ax[0].plot(x_true_fit[:,0], 'c', label='True')
    ax[0].plot(np.array(np.array(x_sim_torch_fit[:,0].detach())), 'r', label='Sim')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(np.array(x_meas_torch_fit[:,1].detach())), 'k*', label='Measured')
    ax[1].plot(np.array(np.array(x_hidden_torch_fit[:,1].detach())), 'b', label='Hidden')
    ax[1].plot(x_true_fit[:,1], 'c', label='True')
    ax[1].plot(np.array(np.array(x_sim_torch_fit[:,1].detach())), 'r', label='Sim')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_fit), label='Input')
    ax[2].grid(True)
        
    # In[Validation]
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
