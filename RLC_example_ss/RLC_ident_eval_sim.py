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
    df_X = pd.read_csv(os.path.join("data", "RLC_data_val.csv"))
#    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))

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

    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_ss_1step_nonoise.pkl")))
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_ss_1step_noise.pkl")))
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_minibatch_128_noise.pkl")))


    # In[Validate model]
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start//Ts)#x.shape[0]
    idx_val_end = int(t_val_end//Ts)#x.shape[0]

    # Build fit data
    u_val = u[idx_val_start:idx_val_end]
    x_meas_val = x_noise[idx_val_start:idx_val_end]
    x_true_val = x[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = time_data[idx_val_start:idx_val_end]

    x_0 = x_meas_val[0, :]

    with torch.no_grad():
        x_sim_torch = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(u_val))
        loss = torch.mean(torch.abs(x_sim_torch - torch.tensor(x_true_val)))


    # In[Plot]
    x_sim = np.array(x_sim_torch)
    fig, ax = plt.subplots(2,1,sharex=True, figsize=(6,5))
    time_val_us = time_val *1e6

    t_plot_start = 0
    t_plot_end = 0.3e-3
    idx_plot_start = int(t_plot_start//Ts)#x.shape[0]
    idx_plot_end = int(t_plot_end//Ts)#x.shape[0]


    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], x_true_val[idx_plot_start:idx_plot_end,0], 'k',  label='True')
    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end,0],'r', label='Model simulation')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time ($\mu$s)")
    ax[0].set_ylabel("Capacitor Voltage (V)")
    ax[0].set_ylim([-400, 400])

    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], np.array(x_true_val[idx_plot_start:idx_plot_end:,1]), 'k', label='True')
    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end:,1],'r', label='Model simulation')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time ($\mu$s)")
    ax[1].set_ylabel("Inductor Current (A)")
    ax[1].set_ylim([-20, 20])

    fig_name = "RLC_SS_val_128step_noise.pdf"
    fig.savefig(fig_name, bbox_inches='tight')
