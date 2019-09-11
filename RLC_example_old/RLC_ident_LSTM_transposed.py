import torch
from torchid.lstmfitter_transposed import LSTMSimulator
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    #y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 1 # 0: voltage 1: current

    y = np.copy(x[:, [y_var_idx]])

    N = np.shape(y)[0]
    Ts = t[1] - t[0]
    t_fit = 2e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 1000
    test_freq = 10

    # set random seed to 0
    #np.random.seed(0)
    #torch.manual_seed(0)


    # build the model
    seq = LSTMSimulator()
    seq.float()
#    seq.double() # useful! cas all to double
    criterion = nn.MSELoss()

    # use LBFGS as optimizer since we can load the whole data to train
    #optimizer = optim.LBFGS(seq.parameters(), lr=0.1)
    optimizer = optim.Adam(seq.parameters(), lr=1e-3)
    #begin to train
    # Batch learning parameters
    seq_len = 64  # int(n_fit/10)
    batch_size = (n_fit - 10) // seq_len
    std_noise_V = 0.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Build fit data
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]
    time_fit = t[0:n_fit]

    def get_batch(batch_size, seq_len):
        num_train_samples = y_fit.shape[0]
        batch_s = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_s[:, np.newaxis] + np.arange(seq_len) # batch all indices
        batch_idx = batch_idx.T
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])#torch.stack([y_meas_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_u = torch.tensor(u_fit[batch_idx]) # torch.stack([u_fit_torch[batch_s[i]:batch_s[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_t = torch.tensor(time_fit[batch_idx])
        return batch_u, batch_y_meas, batch_t

    ii = 0

    time_optim_start = time.time()
    for itr in range(num_iter):
        #print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            batch_u, batch_y_meas, _ = get_batch(batch_size, seq_len)
            batch_y_pred = seq(batch_u)
            loss = criterion(batch_y_pred, batch_y_meas)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        if itr % test_freq == 0:
            with torch.no_grad():
                #y_pred_torch = io_solution.f_onestep(phi_fit_torch) #func(x_true_torch, u_torch)
                #err = y_pred_torch - y_meas_fit_torch[n_max:, :]
                #loss = torch.mean((err) ** 2)  # torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

    time_optimization  = time.time() - time_optim_start

