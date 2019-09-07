import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(".."))
from torchid.iofitter import NeuralIOSimulator
from torchid.iomodels import NeuralIOModel

if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    # y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current

    y = np.copy(x[:, [y_var_idx]])

    N = np.shape(y)[0]
    Ts = t[1] - t[0]
    t_fit = 2e-3
    n_fit = int(t_fit // Ts)  # x.shape[0]
    num_iter = 50000
    test_freq = 100

    n_a = 2  # autoregressive coefficients for y
    n_b = 2  # autoregressive coefficients for u
    n_max = np.max((n_a, n_b))  # delay

    # Batch learning parameters
    seq_len = 16  # int(n_fit/10)
    batch_size = (n_fit - n_a) // seq_len

    std_noise_V = 0.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Initialize optimization
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64)
    io_solution = NeuralIOSimulator(io_model)
    io_solution.io_model.load_state_dict(torch.load(os.path.join("models", "model_IO_128step_noise.pkl")))

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

