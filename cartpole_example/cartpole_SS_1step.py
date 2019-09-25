import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(".."))
from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.ssmodels import MechanicalStateSpaceModel

# In[Load data]
if __name__ == '__main__':

    add_noise = False

    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    COL_R = ['r']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC_ref.csv"))
    #df_X = pd.read_csv(os.path.join("data", "pendulum_data_PID.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = t[1] - t[0]
    x_noise = x
 
# In[Model]
    ss_model = MechanicalStateSpaceModel(Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
   
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
    test_freq = 100

    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    end = time.time()

    with torch.no_grad():
        x_est_torch = nn_solution.f_onestep(x_meas_fit_torch, u_fit_torch)
        err_init = x_est_torch - x_meas_fit_torch
        scale_error = torch.sqrt(torch.mean((err_init)**2, dim=0)) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))
        #scale_error[0] = 0.0
        #scale_error[3] = 0.0

    #scale_error = 1./np.std(np.diff(x_fit, axis = 0), axis=0)
    #scale_error = torch.tensor(scale_error.astype(np.float32))

# In[Fit model]
    LOSS = []
    time_start = time.time()
    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()
        x_pred_torch = nn_solution.f_onestep(x_meas_fit_torch, u_fit_torch)
        err = x_pred_torch - x_meas_fit_torch
        err_scaled = err / scale_error
        loss_sc = torch.mean((err_scaled[:,[1,3]]) ** 2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))

        LOSS.append(loss_sc.item())
        loss_sc.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss_sc.item()))
                ii += 1
        end = time.time()

# In[Save model]
    if not os.path.exists("models"):
        os.makedirs("models")

    model_name = "model_SS_1step_nonoise.pkl"
    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_name))



# In[Simulate model]
    x_0 = x_fit[0, :]
    with torch.no_grad():
        x_sim_torch = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(u_fit))
        loss_sc = torch.mean(torch.abs(x_sim_torch - x_meas_fit_torch))
        x_sim = np.array(x_sim_torch)
    # In[1]
    n_plot = 150
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


    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = "cartpole_SS_loss_1step_noise.pdf"
    else:
        fig_name = "cartpole_SS_loss_1step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
