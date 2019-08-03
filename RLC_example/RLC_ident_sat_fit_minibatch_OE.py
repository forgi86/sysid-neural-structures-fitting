import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt


from symbolic_RLC import fxu_ODE, fxu_ODE_mod
from neuralode import  NeuralODE, RunningAverageMeter


if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv("RLC_data_sat_FE.csv")
    #df_X = pd.read_csv("RLC_data.csv")
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])

    Ts = t[1] - t[0]
    t_fit = 5e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 1000
    seq_len = 400
    batch_size = 128
    test_freq = 10

    # Get fit data #
    u_fit = u[0:n_fit]
    x_fit = x[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    y_true_torch_fit = torch.from_numpy(y_fit)
    x_true_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    def get_batch(batch_size, seq_len):
        num_train_samples = x_true_torch_fit.shape[0]
        s = torch.from_numpy(np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False))
        batch_x0 = x_true_torch_fit[s, :]  # (M, D)
        batch_t = torch.stack([time_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.stack([x_true_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_u = torch.stack([u_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        
        return batch_t, batch_x0, batch_u, batch_x 
    
    
    nn_solution = NeuralODE()
    nn_solution.load_state_dict(torch.load('model_ARX_FE_sat.pkl'))

    optimizer = optim.Adam(nn_solution.parameters(), lr=1e-5)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)



    ii = 0
    for itr in range(0, num_iter):


        if itr % test_freq == 0:
            with torch.no_grad():
                x_pred_torch = nn_solution.f_OE(x0_torch, u_torch_fit)
                loss = torch.mean(torch.abs(x_pred_torch - x_true_torch_fit))
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

        batch_x_pred = nn_solution.f_OE_minibatch(batch_x0, batch_u)
#        err = torch.abs(batch_x[:, 1:, :] - batch_x_pred[:, 1:, :])
#        err[:,:,1] = err[:,:,1]*100.0
#        loss = torch.mean(err)
        loss = torch.mean((batch_x[:,1:,:] - batch_x_pred[:,1:,:])**2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())


        end = time.time()

    #torch.save(nn_solution.state_dict(), 'model.pkl')

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
