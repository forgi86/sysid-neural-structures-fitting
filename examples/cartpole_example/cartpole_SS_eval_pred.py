import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join("..", ".."))
from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.ssmodels import CartPoleStateSpaceModel
from torchid.util import get_sequential_batch_idx

if __name__ == '__main__':

    seq_len = 512  # simulation sequence length - we evaluate performance in terms of seq_len-step simulation error
    dataset_filename = "pendulum_data_oloop_val.csv"
    model_filename = "model_SS_64step_noise.pkl"

    # Column names in the dataset
    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", dataset_filename), sep=",")
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    x0_torch = torch.from_numpy(x[0, :])

    N = np.shape(y)[0]
    Ts = time_data[1] - time_data[0]

    # Load model and parameters
    ss_model = CartPoleStateSpaceModel(Ts, init_small=True)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))


    # Build validation data
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start//Ts)
    idx_val_end = int(t_val_end//Ts)
    u_val = u[idx_val_start:idx_val_end]
    x_val = x[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = time_data[idx_val_start:idx_val_end]

    # Predict batch data
    batch_start, batch_idx = get_sequential_batch_idx(y_val.shape[0], seq_len)
    batch_time = torch.tensor(time_val[batch_idx])
    batch_x0 = torch.tensor(x_val[batch_start])
    batch_u = torch.tensor(u_val[batch_idx])
    batch_x = torch.tensor(x_val[batch_idx])
    batch_x_pred = nn_solution.f_sim_multistep(batch_x0, batch_u)

    # Plot data
    batch_x_pred_np = np.array(batch_x_pred.detach())
    batch_time_np = np.array(batch_time.detach()).squeeze()

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(time_val, x_val[:, 0], 'b')
    ax[0].plot(batch_time_np.T, batch_x_pred_np[:, :, 0].T, 'r')
    ax[0].grid(True)

    ax[1].plot(time_val, x_val[:,2], 'b')
    ax[1].plot(batch_time_np.T, batch_x_pred_np[:, :, 2].T, 'r')
    ax[1].grid(True)

    ax[2].plot(time_val, u_val, label='Input')
    ax[2].grid(True)
