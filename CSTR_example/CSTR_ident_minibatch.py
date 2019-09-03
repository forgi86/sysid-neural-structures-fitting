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

from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.util import RunningAverageMeter
from torchid.ssmodels import NeuralStateSpaceModelLin, NeuralStateSpaceModel


if __name__ == '__main__':

    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    df_X = pd.read_csv(os.path.join("data", "cstr.dat"), header=None, sep="\t")
    df_X.columns = ['time', 'q', 'Ca', 'T', 'None']

    df_X['q'] = df_X['q']/100
    df_X['Ca'] = df_X['Ca']*10
    df_X['T'] = df_X['T']/400
    
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])


    x_noise = np.copy(x) #+ np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)

    Ts = time_data[1] - time_data[0]
    t_fit = time_data[-1]
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 20000
    test_freq = 100    
    seq_len = 64 #int(n_fit/10)
    batch_size = n_fit//seq_len

    # Get fit data #
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = time_data[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    y_true_torch_fit = torch.from_numpy(y_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    def get_batch(batch_size, seq_len):
        num_train_samples = x_meas_torch_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False)
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch all indices

        batch_t = torch.tensor(time_fit[batch_idx]) #torch.stack([time_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x0 = torch.tensor(x_fit[batch_start]) #x_meas_torch_fit[batch_start, :]  # (M, D)
        batch_u = torch.tensor(u_fit[batch_idx]) #torch.stack([u_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.tensor(x_fit[batch_idx]) #torch.stack([x_meas_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)

        return batch_t, batch_x0, batch_u, batch_x 
    

#    ss_model = NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64) #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_ss_64.pkl")))

    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)


    with torch.no_grad():
        batch_t, batch_x0, batch_u, batch_x = get_batch(batch_size, seq_len)
        batch_x_pred = nn_solution.f_sim_minibatch(batch_x0, batch_u)    
        err = batch_x[:,0:,:] - batch_x_pred[:,0:,:]
        loss_scale = torch.mean(err**2)
    
        
    ii = 0
    for itr in range(0, num_iter):


        if  itr > 0 and itr % test_freq == 0:
            with torch.no_grad():
                #x_pred_torch = nn_solution.f_OE(x0_torch, u_torch_fit)
                #loss = torch.mean(torch.abs(x_pred_torch - x_meas_torch_fit))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        optimizer.zero_grad()
        batch_t, batch_x0, batch_u, batch_x = get_batch(batch_size, seq_len)
        #batch_size = 256
        #N = x_true_torch_fit.shape[0]
        #N = int(N // batch_size) * batch_size
        #seq_len = int(N // batch_size)
        #batch_x = x_true_torch_fit[0:N].view(batch_size, seq_len, -1)
        #batch_u = u_torch_fit[0:N].view(batch_size, seq_len, -1)
        #batch_x0 = batch_x[:, 0, :]

        batch_x_pred = nn_solution.f_sim_minibatch(batch_x0, batch_u)
#        err = torch.abs(batch_x[:, 1:, :] - batch_x_pred[:, 1:, :])
#        err[:,:,1] = err[:,:,1]*100.0
#        loss = torch.mean(err)
        err = batch_x[:,0:,:] - batch_x_pred[:,0:,:]
        err_scaled = err  
        loss = torch.mean(err_scaled**2)/loss_scale
#        loss = torch.mean((batch_x[:,1:,:] - batch_x_pred[:,1:,:])**2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())


        end = time.time()

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", "model_ss_64.pkl"))

    # In[Validate model]
    t_val = time_data[-1]
    n_val = int(t_val//Ts)#x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]
    output_data_val = y[0:n_val]

    x0_val = state_data_val[0,:]#np.zeros(2,dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)

    with torch.no_grad():
        x_pred_torch_val = nn_solution.f_sim(x0_torch_val, u_torch_val)

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
