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

    len_fit = 40
    seq_len = 50
    test_freq = 20
    num_iter = 2000
    test_freq = 1
    add_noise = False

    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC_ref.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = t[1] - t[0]
    x_noise = x
 
    n_x = x.shape[-1]
    ss_model = MechanicalStateSpaceModel(Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    #model_name = "model_SS_1step_nonoise.pkl"
    #model_name = "model_SS_150step_nonoise.pkl"
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_name )))

    n_fit = int(len_fit//Ts)
    time_fit = t[0:n_fit]
    u_fit = u[0:n_fit]
    x_meas_fit = x_noise[0:n_fit]
    y_meas_fit = y[0:n_fit]
    batch_size = n_fit//seq_len

    x_hidden_fit = torch.tensor(x_meas_fit, requires_grad=True)  # this is an optimization variable!

    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=1e-5)
    end = time.time()


# In[Batch function]

    def get_batch(batch_size, seq_len):
        num_train_samples = x_meas_fit.shape[0]
        batch_start = batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False)
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch all indices

        batch_x0 = torch.tensor(x_meas_fit[batch_start, :])
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx]) #torch.stack([u_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.tensor(x_meas_fit[batch_idx]) #torch.stack([x_meas_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        
        return batch_t, batch_x0, batch_u, batch_x



    def get_sequential_batch(seq_len):
        num_train_samples = x_meas_fit.shape[0]
        batch_size = num_train_samples//seq_len
        batch_start = np.arange(0, batch_size, dtype=np.int64) * seq_len
        batch_idx = batch_start[:,np.newaxis] + np.arange(seq_len) # batch all indices

        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        return batch_t, batch_x0_hidden, batch_u, batch_y_meas


    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_y_meas = get_sequential_batch(seq_len)
        batch_x_sim = nn_solution.f_sim_minibatch(batch_x0_hidden, batch_u)
        err_init = batch_x_sim[:, :, [0, 2]] - batch_y_meas
        scale_error = torch.sqrt(torch.mean((err_init)**2,dim=(0,1)))
        
# In[Fit model]
    LOSS = []
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        batch_t, batch_x0_hidden, batch_u, batch_y_meas = get_sequential_batch(seq_len)
        batch_x_sim = nn_solution.f_sim_minibatch(batch_x0_hidden, batch_u)
        err = batch_x_sim[:, :, [0, 2]] - batch_y_meas
        err_scaled = err/scale_error
        loss = torch.mean(err_scaled**2)

        LOSS.append(loss.item())

        if itr > 0 and itr % test_freq == 0:
            with torch.no_grad():
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        loss.backward()
        optimizer.step()

        end = time.time()

    # In[Save model parameters]
#    model_name = "model_SS_150step_nonoise.pkl"
    model_name = "model_SS_200step_nonoise.pkl"

    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_name))    
    
# In[Simulate model]
    x_0 = x_meas_fit[0, :]
    with torch.no_grad():
        x_meas_fit_torch = torch.tensor(x_meas_fit)
        x_sim_torch = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(u_fit))
        loss = torch.mean(torch.abs(x_sim_torch - x_meas_fit_torch))
        x_sim = np.array(x_sim_torch)
    # In[1]
    n_plot = 50

    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(time_fit[:n_plot], x_meas_fit[:n_plot, 0], label='True')
    ax[0].plot(time_fit[:n_plot], x_sim[:n_plot, 0], label='Simulated')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(time_fit[:n_plot], x[:n_plot, 2], label='True')
    ax[1].plot(time_fit[:n_plot], x_sim[:n_plot, 2], label='Simulated')
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
        fig_name = f"cartpole_SS_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"cartpole_SS_loss_{seq_len}step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
