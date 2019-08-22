import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(".."))
from torchid.ssfitter import  NeuralODE, RunningAverageMeter
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
    model_name = "model_ARX_FE_sat_nonoise.pkl"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_name)))
# In[Setup optimization problem]

    len_fit = 40
    n_fit = int(len_fit//Ts)
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    t_fit = t[0:n_fit]
    u_fit_torch = torch.from_numpy(u_fit)
    x_meas_fit_torch = torch.from_numpy(x_fit)
    t_fit_torch = torch.from_numpy(t_fit)
    
    num_iter = 10000
    test_freq = 1

    params = nn_solution.ss_model.parameters()
    optimizer = optim.Adam(params, lr=1e-4)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    
    #scale_error = 1./np.std(x_noise, axis=0)
    #scale_error = scale_error/np.sum(scale_error)


# In[Batch function]
    seq_len = 100 #int(n_fit/10)
    batch_size = n_fit//seq_len
    test_freq = 10
    def get_batch(batch_size, seq_len):
        num_train_samples = x_meas_fit_torch.shape[0]
        s = torch.from_numpy(np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False))
        batch_x0 = x_meas_fit_torch[s, :]  # (M, D)
        batch_t = torch.stack([t_fit_torch[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.stack([x_meas_fit_torch[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_u = torch.stack([u_fit_torch[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        
        return batch_t, batch_x0, batch_u, batch_x 
# In[Scale]
    scale_error = 1e2*np.ones(4)/4
    scale_error = torch.tensor(scale_error.astype(np.float32))        
# In[Fit model]
    ii = 0
    loss = None
    for itr in range(0, num_iter):


        if  itr > 0 and itr % test_freq == 0:
            with torch.no_grad():
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
        err = batch_x[:,0:,:] - batch_x_pred[:,0:,:]
        err_scaled = err * scale_error        
        loss = torch.mean(err_scaled**2)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        end = time.time()

# In[Save model parameters]
        
# In[Simulate model]
    x_0 = x_fit[0, :]
    with torch.no_grad():
        x_sim_torch = nn_solution.f_OE(torch.tensor(x_0), torch.tensor(u_fit))
        loss = torch.mean(torch.abs(x_sim_torch - x_meas_fit_torch))
        x_sim = np.array(x_sim_torch)
    # In[1]
    n_plot = 4000

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
