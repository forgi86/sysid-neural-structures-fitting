import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(".."))
from torchid.iofitter import NeuralIOSimulator
from torchid.iomodels import NeuralIOModel
import scipy.linalg
from torchid.util import get_random_batch_idx, get_sequential_batch_idx

if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    # y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current

    y = np.copy(x[:, [y_var_idx]])

    N = np.shape(y)[0]
    Ts = time_data[1] - time_data[0]

    n_a = 2  # autoregressive coefficients for y
    n_b = 2  # autoregressive coefficients for u
    n_max = np.max((n_a, n_b))  # delay

    std_noise_V = 0.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Initialize optimization
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64)
    io_solution = NeuralIOSimulator(io_model)
    io_solution.io_model.load_state_dict(torch.load(os.path.join("models", "model_IO_1step_nonoise.pkl")))

    # In[Validate model]
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start//Ts)#x.shape[0]
    idx_val_end = int(t_val_end//Ts)#x.shape[0]

    # Build fit data
    time_val = time_data[idx_val_start:idx_val_end]
    u_val = u[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    y_meas_val = y_noise[idx_val_start:idx_val_end]
    phi_val_y = scipy.linalg.toeplitz(y_meas_val, y_meas_val[0:n_a])[n_max - 1:-1, :] # regressor 1
    phi_val_u = scipy.linalg.toeplitz(u_val, u_val[0:n_a])[n_max - 1:-1, :]
    phi_val = np.hstack((phi_val_y, phi_val_u))

    # Neglect initial values
    y_val = y_val[n_max:,:]
    y_meas_val = y_meas_val[n_max:,:]
    u_val = u_val[n_max:, :]
    time_val = time_val[n_max:, :]

    # Predict batch data
    seq_len = 128
    batch_start, batch_idx = get_sequential_batch_idx(y_meas_val.shape[0], seq_len)
    batch_y_seq = torch.tensor(phi_val_y[batch_start])
    batch_u_seq = torch.tensor(phi_val_u[batch_start])
    batch_y_meas = torch.tensor(y_meas_val[batch_idx]) #torch.stack([y_meas_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_u = torch.tensor(u_val[batch_idx]) # torch.stack([u_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_time = torch.tensor(time_val[batch_idx])
    batch_y_pred = io_solution.f_sim_minibatch(batch_u, batch_y_seq, batch_u_seq)


    # Plot data
    batch_y_meas_np = np.array(batch_y_meas.detach()).squeeze()
    batch_y_pred_np = np.array(batch_y_pred.detach()).squeeze()
    batch_time_np = np.array(batch_time.detach()).squeeze()

    plt.plot(time_val, y_val, 'b')
    plt.plot(batch_time_np.T, batch_y_pred_np.T, 'r')
