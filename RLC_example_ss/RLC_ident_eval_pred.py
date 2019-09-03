import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(".."))
from torchid.ssfitter import NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel
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
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64) #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_ARX_FE_sat_nonoise.pkl")))


    # In[Validate model]
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start//Ts)#x.shape[0]
    idx_val_end = int(t_val_end//Ts)#x.shape[0]

    # Build fit data
    u_val = u[idx_val_start:idx_val_end]
    x_val = x_noise[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = time_data[idx_val_start:idx_val_end]


    # Predict batch data
    seq_len = 128
    batch_start, batch_idx = get_sequential_batch_idx(y_val.shape[0], seq_len)
    batch_time = torch.tensor(time_val[batch_idx])  # torch.stack([time_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_x0 = torch.tensor(x_val[batch_start])  # x_meas_torch_fit[batch_start, :]  # (M, D)
    batch_u = torch.tensor(u_val[batch_idx])  # torch.stack([u_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_x = torch.tensor(x_val[batch_idx])  # torch.stack([x_meas_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_x_pred = nn_solution.f_sim_minibatch(batch_x0, batch_u)

    # Plot data
    batch_x_pred_np = np.array(batch_x_pred.detach())
    batch_time_np = np.array(batch_time.detach()).squeeze()

    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(time_val, x_val[:,0], 'b')
    ax[0].plot(batch_time_np.T, batch_x_pred_np[:,:,0].T, 'r')
    ax[0].grid(True)

    ax[1].plot(time_val, x_val[:,1], 'b')
    ax[1].plot(batch_time_np.T, batch_x_pred_np[:,:,1].T, 'r')
    ax[1].grid(True)

    ax[2].plot(time_val, u_val, label='Input')
    ax[2].grid(True)
