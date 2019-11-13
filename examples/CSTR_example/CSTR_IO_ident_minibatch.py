import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.linalg

sys.path.append(os.path.join(".."))
from torchid.iofitter import NeuralIOSimulator
from torchid.util import RunningAverageMeter
from torchid.iomodels import NeuralIOModel


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    n_a = 2 # autoregressive coefficients for y
    n_b = 2 # autoregressive coefficients for u
    seq_len = 32  # subsequence length m
    lr = 1e-4
    num_iter = 50000
    test_freq = 100

    # Column names
    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "CSTR_data_id.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0 # 0: Ca 1: T
    y = np.copy(x[:, [y_var_idx]])
    N = np.shape(y)[0]
    Ts = t[1] - t[0]

    # Add noise - no noise here
    x_noise = np.copy(x)
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:,[y_var_idx]]

    # Get fit data
    t_fit = t[-1]
    n_fit = int(t_fit//Ts)#x.shape[0]
    n_max = np.max((n_a, n_b)) # delay
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]
    phi_fit_y = scipy.linalg.toeplitz(y_meas_fit, y_meas_fit[0:n_a])[n_max - 1:-1, :] # regressor 1
    phi_fit_u = scipy.linalg.toeplitz(u_fit, u_fit[0:n_a])[n_max - 1:-1, :]
    phi_fit = np.hstack((phi_fit_y, phi_fit_u))

    # Batch learning parameters
    batch_size = (n_fit - n_a) // seq_len


    # Neglect initial values
    y_fit = y_fit[n_max:,:]
    y_meas_fit = y_meas_fit[n_max:,:]
    u_fit = u_fit[n_max:, :]

    # To Pytorch tensors
    phi_fit_torch = torch.from_numpy(phi_fit)
    y_meas_fit_torch = torch.from_numpy(y_meas_fit)
    u_fit_torch = torch.from_numpy(u_fit)
    phi_fit_y_torch = torch.tensor(phi_fit_y)
    phi_fit_u_torch = torch.tensor(phi_fit_u)

    # Setup neural model structure
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64)
    io_solution = NeuralIOSimulator(io_model)

    # Setup optimizer
    optimizer = optim.Adam(io_solution.io_model.parameters(), lr=lr)

    def get_batch(batch_size, seq_len):
        num_train_samples = y_meas_fit_torch.shape[0]
        batch_s = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_s[:,np.newaxis] + np.arange(seq_len) # batch all indices
        batch_y_seq = phi_fit_y_torch[batch_s]
        batch_u_seq = phi_fit_u_torch[batch_s]
        #batch_t = torch.stack([time_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx]) #torch.stack([y_meas_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_u = torch.tensor(u_fit[batch_idx]) # torch.stack([u_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)

        return batch_u, batch_y_meas, batch_y_seq, batch_u_seq, batch_s


    def get_sequential_batch(seq_len):
        num_train_samples = y_meas_fit_torch.shape[0]
        batch_size = num_train_samples//seq_len
        batch_s = np.arange(0, batch_size, dtype=np.int64) * seq_len
        batch_idx = batch_s[:,np.newaxis] + np.arange(seq_len) # batch all indices
        batch_y_seq = phi_fit_y_torch[batch_s]
        batch_u_seq = phi_fit_u_torch[batch_s]
        #batch_t = torch.stack([time_torch_fit[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx]) #torch.stack([y_meas_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_u = torch.tensor(u_fit[batch_idx]) # torch.stack([u_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)

        return batch_u, batch_y_meas, batch_y_seq, batch_u_seq, batch_s

    with torch.no_grad():
        batch_u, batch_y_meas, batch_y_seq, batch_u_seq, batch_s = get_batch(batch_size, seq_len)
        batch_y_pred = io_solution.f_sim_multistep(batch_u, batch_y_seq, batch_u_seq)
        err = batch_y_meas[:, 0:, :] - batch_y_pred[:, 0:, :]
        loss = torch.mean((err) ** 2)
        loss_scale = np.float32(loss)

    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Predict
        batch_u, batch_y_meas, batch_y_seq, batch_u_seq, batch_s = get_batch(batch_size, seq_len)
        batch_y_pred = io_solution.f_sim_multistep(batch_u, batch_y_seq, batch_u_seq)

        # Compute loss
        err = batch_y_meas[:,0:,:] - batch_y_pred[:,0:,:]
        loss = torch.mean((err)**2)/loss_scale

        # Statistics
        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # Optimization step
        loss.backward()
        optimizer.step()



        LOSS.append(loss.item())

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save fitted model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(io_solution.io_model.state_dict(), os.path.join("models", "model_IO_32step.pkl"))


    # In[Validate model]
    t_val_start = 0
    t_val_end = t[-1]
    idx_val_start = int(t_val_start//Ts)#x.shape[0]
    idx_val_end = int(t_val_end//Ts)#x.shape[0]

    n_val = idx_val_end - idx_val_start
    u_val = np.copy(u[idx_val_start:idx_val_end])
    y_val = np.copy(y[idx_val_start:idx_val_end])
    y_meas_val = np.copy(y_noise[idx_val_start:idx_val_end])

    y_seq = np.array(np.flip(y_val[0:n_a].ravel()))
    u_seq = np.array(np.flip(u_val[0:n_b].ravel()))

    # Neglect initial values
    y_val = y_val[n_max:,:]
    y_meas_val = y_meas_val[n_max:,:]
    u_val = u_val[n_max:, :]

    y_meas_val_torch = torch.tensor(y_meas_val)

    with torch.no_grad():
        y_seq_torch = torch.tensor(y_seq)
        u_seq_torch = torch.tensor(u_seq)

        u_torch = torch.tensor(u_val)
        y_val_sim_torch = io_solution.f_sim(y_seq_torch, u_seq_torch, u_torch)

        err_val = y_val_sim_torch - y_meas_val_torch
        loss_val =  torch.mean((err_val)**2)

    # In[Plot]
    y_val_sim = np.array(y_val_sim_torch)
    fig,ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(y_val, 'b', label='True')
    ax[0].plot(y_val_sim, 'r',  label='Sim')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(u_val, label='Input')
    ax[1].legend()
    ax[1].grid(True)


    fig,ax = plt.subplots(1,1)
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Itaration")        
