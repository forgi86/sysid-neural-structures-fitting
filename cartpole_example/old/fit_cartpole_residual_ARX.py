import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(".."))
from torchid.ssfitter import  NeuralStateSpaceSimulator, NeuralSumODE, RunningAverageMeter
from torchid.ssmodels import MechanicalStateSpaceModel, NeuralStateSpaceModel

# In[Load data]
if __name__ == '__main__':
    
    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    COL_R = ['r']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC_ref.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    #u = np.array(df_X[COL_U],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = t[1] - t[0]
    x_noise = x
 
# In[Model]
    ss_model = MechanicalStateSpaceModel(Ts)
    model_name = "model_ARX_FE_nonoise.pkl"
    ss_model.load_state_dict(torch.load(os.path.join("models", model_name)))   

    ss_model_residual = NeuralStateSpaceModel(n_x=4, n_u=1, n_feat=64)
    nn_solution = NeuralSumODE([ss_model,ss_model_residual])
   
# In[Setup optimization problem]

    len_fit = 40
    n_fit = int(len_fit//Ts)
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    t_fit = t[0:n_fit]
    u_fit_torch = torch.from_numpy(u_fit)
    x_meas_fit_torch = torch.from_numpy(x_fit)
    t_fit_torch = torch.from_numpy(t_fit)
    
    num_iter = 20000
    test_freq = 1

    params = list(nn_solution.ss_model_list[1].parameters())
    optimizer = optim.Adam(params, lr=1e-5)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    
    #scale_error = 1./np.std(x_noise, axis=0)
    #scale_error = scale_error/np.sum(scale_error)
    #scale_error = 1e0*np.ones(4)/4
    #scale_error = 1./np.mean(np.abs(np.diff(x_fit, axis = 0)), axis=0)
    scale_error = 1./np.std(np.diff(x_fit, axis = 0), axis=0)
    scale_error = torch.tensor(scale_error.astype(np.float32))

# In[Fit model]
    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()
        x_pred_torch = nn_solution.f_ARX(x_meas_fit_torch, u_fit_torch)
        err = x_pred_torch - x_meas_fit_torch
        err_scaled = err * scale_error
        loss = 10e3*torch.mean((err_scaled)**2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % test_freq == 0:
            with torch.no_grad():
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
        end = time.time()




# In[Simulate model]
    x_0 = x_fit[0, :]
    with torch.no_grad():
        x_sim_torch = nn_solution.f_OE(torch.tensor(x_0), torch.tensor(u_fit))
        loss = torch.mean(torch.abs(x_sim_torch - x_meas_fit_torch))
        x_sim = np.array(x_sim_torch)
    # In[1]
    n_plot = 200
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(t_fit[:n_plot], x_fit[:n_plot, 0], label='True')
    ax[0].plot(t_fit[:n_plot], x_sim[:n_plot,0], label='Simulated')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(t_fit[:n_plot], x[:n_plot,2], label='True')
    ax[1].plot(t_fit[:n_plot], x_sim[:n_plot, 2], label='Simulated')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle (rad)")
    ax[1].legend()
    ax[1].grid()
