import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import pickle
from torchid.ssfitter import NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel
from utils import metrics

if __name__ == '__main__':

    dataset_type = 'val'
    model_type = '1step_nonoise'

    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    dataset_filename = f"CSTR_data_{dataset_type}.csv"
    df_X = pd.read_csv(os.path.join("data", dataset_filename))

    with open(os.path.join("data", "fit_scaler.pkl"), 'rb') as fp:
        scaler = pickle.load(fp)

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current

    y = np.copy(x[:, [y_var_idx]])

    N = np.shape(y)[0]
    Ts = time_data[1] - time_data[0]

    x_noise = np.copy(x)
    x_noise = x_noise.astype(np.float32)

    # Initialize optimization
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64) #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    model_filename = f"model_SS_{model_type}.pkl"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))

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

    if dataset_type == 'id':
        t_plot_start = 0.0
        t_plot_end = 500
    else:
        t_plot_start = 0.0
        t_plot_end = 100

    idx_plot_start = int(t_plot_start//Ts)#x.shape[0]
    idx_plot_end = int(t_plot_end//Ts)#x.shape[0]


    x_true_val_unsc = np.copy(x_true_val)
    x_true_val_unsc[:, 0] = x_true_val[:,0]*scaler.scale_[0] + scaler.mean_[0]
    x_true_val_unsc[:, 1] = x_true_val[:, 1]*scaler.scale_[1] + scaler.mean_[1]

    x_sim_unsc = np.copy(x_sim)
    x_sim_unsc[:, 0] = x_sim_unsc[:,0]*scaler.scale_[0] + scaler.mean_[0]
    x_sim_unsc[:, 1] = x_sim_unsc[:, 1]*scaler.scale_[1] + scaler.mean_[1]


    ax[0].plot(time_val[idx_plot_start:idx_plot_end], x_true_val_unsc[idx_plot_start:idx_plot_end,0], 'k',  label='True')
    ax[0].plot(time_val[idx_plot_start:idx_plot_end], x_sim_unsc[idx_plot_start:idx_plot_end,0],'r--', label='Model simulation')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Concentration Ca (mol/L)")
#    ax[0].set_ylim([-400, 400])

    ax[1].plot(time_val[idx_plot_start:idx_plot_end],  x_true_val_unsc[idx_plot_start:idx_plot_end,1], 'k', label='True')
    ax[1].plot(time_val[idx_plot_start:idx_plot_end], x_sim_unsc[idx_plot_start:idx_plot_end:,1],'r--', label='Model simulation')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Reactor temperature T (K)")
#    ax[1].set_ylim([-20, 20])

    fig_name = f"CSTR_SS_{dataset_type}_{model_type}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
