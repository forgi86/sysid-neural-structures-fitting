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
from torchid.util import get_torch_regressor_mat

if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    N = np.shape(y)[0]
    Ts = t[1] - t[0]
    t_fit = 2e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 20000
    test_freq = 100

    n_a = 2 # autoregressive coefficients for y
    n_b = 2 # autoregressive coefficients for u
    n_max = np.max((n_a, n_b)) # delay

    # Batch learning parameters
    seq_len = 32  # int(n_fit/10)
    batch_size = (n_fit - n_a) // seq_len

    std_noise_V = 1.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:,[0]]

    # Build fit data
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]

    h_fit = np.copy(y_meas_fit)
    h_fit = np.vstack((np.zeros(n_a).reshape(-1, 1), h_fit))
    v_fit = np.copy(u_fit)
    v_fit = np.vstack((np.zeros(n_b).reshape(-1, 1), v_fit))

    phi_fit_y = scipy.linalg.toeplitz(h_fit, h_fit[0:n_a])[n_max - 1:-1, :] # regressor 1
    phi_fit_u = scipy.linalg.toeplitz(v_fit, v_fit[0:n_a])[n_max - 1:-1, :]
    phi_fit = np.hstack((phi_fit_y, phi_fit_u))

    # To Pytorch tensors
    h_fit_torch = torch.tensor(h_fit, requires_grad=True) # this is an optimization variable!
    phi_fit_u_torch = torch.tensor(phi_fit_u)
    phi_fit_y_torch = get_torch_regressor_mat(h_fit_torch.view(-1), n_a)
    y_meas_fit_torch = torch.tensor(y_meas_fit)




