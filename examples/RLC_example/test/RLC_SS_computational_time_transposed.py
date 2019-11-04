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
import time

if __name__ == '__main__':
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    # df_X = pd.read_csv("RLC_data.csv")
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    x0_torch = torch.from_numpy(x[0, :])

    std_noise_V = 10.0
    std_noise_I = 1.0
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    x0_torch = torch.from_numpy(x[0, :])

    N = np.shape(y)[0]
    Ts = time_data[1] - time_data[0]

    std_noise_V = 0.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = np.copy(y)

    # Initialize optimization
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)  # NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models",  "model_SS_1step_nonoise.pkl")))

    # In[Validate model]
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start // Ts)  # x.shape[0]
    idx_val_end = int(t_val_end // Ts)  # x.shape[0]

    # Build fit data
    u_val = u[idx_val_start:idx_val_end]
    x_val = x_noise[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = time_data[idx_val_start:idx_val_end]


    # Predict batch data
    SEQ_LEN = np.concatenate([np.arange(8, 128), np.arange(128, 4000, 200)])

    TIME_CALC_NOGRAD = []
    TIME_CALC_GRAD = []

    num_samples = y_val.shape[0]

    loss_fn = torch.nn.MSELoss()


    for idx in range(len(SEQ_LEN)):
        seq_len = SEQ_LEN[idx]
        batch_size = num_samples // seq_len
        batch_start, batch_idx = get_random_batch_idx(num_samples, batch_size, seq_len)
        batch_idx = batch_idx.T
        batch_time = torch.tensor(time_val[batch_idx])  # torch.stack([time_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x0 = torch.tensor(x_val[batch_start])  # x_meas_torch_fit[batch_start, :]  # (M, D)
        batch_u = torch.tensor(u_val[batch_idx])  # torch.stack([u_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.tensor(x_val[batch_idx])  # torch.stack([x_meas_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        time_start = time.perf_counter()

        with torch.no_grad():
            batch_x_pred = nn_solution.f_sim_minibatch_transposed(batch_x0, batch_u)
            err = batch_x - batch_x_pred
            err_scaled = err
            loss = torch.mean(err_scaled**2)
            val = (time.perf_counter() - time_start) #/ (seq_len * batch_size)
            TIME_CALC_NOGRAD.append(val)


    for idx in range(len(SEQ_LEN)):
        seq_len = SEQ_LEN[idx]
        batch_size = num_samples // seq_len
        batch_start, batch_idx = get_random_batch_idx(num_samples, batch_size, seq_len)
        batch_idx = batch_idx.T
        batch_time = torch.tensor(time_val[batch_idx])  # torch.stack([time_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x0 = torch.tensor(x_val[batch_start])  # x_meas_torch_fit[batch_start, :]  # (M, D)
        batch_u = torch.tensor(u_val[batch_idx])  # torch.stack([u_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.tensor(x_val[batch_idx])  # torch.stack([x_meas_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        time_start = time.perf_counter()

        time_start = time.perf_counter()

        batch_x_pred = nn_solution.f_sim_minibatch_transposed(batch_x0, batch_u)
        err = batch_x - batch_x_pred
        err_scaled = err
        loss = torch.mean(err_scaled ** 2)
        # loss = loss_fn(batch_x_pred, batch_x)
        loss.backward()
        val = (time.perf_counter() - time_start) #/ (seq_len * batch_size)
        TIME_CALC_GRAD.append(val)

        for par in nn_solution.ss_model.parameters():
            if par.grad is not None:
                par.grad.zero_()


    # In[Plot]
    plt.plot(SEQ_LEN, np.array(TIME_CALC_NOGRAD)*1e3, '*b', label='Forward pass only')
    plt.plot(SEQ_LEN, np.array(TIME_CALC_GRAD) * 1e3, '*r', label='Forward and backward pass')
    plt.legend()
    plt.grid(True)
