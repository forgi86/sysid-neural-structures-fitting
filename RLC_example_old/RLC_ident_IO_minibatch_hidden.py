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
    num_iter = 30000 # 40000
    test_freq = 100

    n_a = 2 # autoregressive coefficients for y
    n_b = 2 # autoregressive coefficients for u
    n_max = np.max((n_a, n_b)) # delay

    # Batch learning parameters
    seq_len = 128  # int(n_fit/10)
    batch_size = (n_fit - n_a) // seq_len

    std_noise_V = 1.0 * 10.0
    std_noise_I = 1.0 * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:,[0]]

    # Build fit data
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]

    h_fit = np.copy(y_meas_fit)
    h_fit = np.vstack((np.zeros(n_a).reshape(-1, 1), h_fit)).astype(np.float32)
    v_fit = np.copy(u_fit)
    v_fit = np.vstack((np.zeros(n_b).reshape(-1, 1), v_fit)).astype(np.float32)

    phi_fit_y = scipy.linalg.toeplitz(h_fit, h_fit[0:n_a])[n_max - 1:-1, :] # regressor 1
    phi_fit_u = scipy.linalg.toeplitz(v_fit, v_fit[0:n_a])[n_max - 1:-1, :]
    phi_fit = np.hstack((phi_fit_y, phi_fit_u))

    # To pytorch tensors
    phi_fit_u_torch = torch.tensor(phi_fit_u)
    h_fit_torch = torch.tensor(h_fit, requires_grad=True) # this is an optimization variable!
    phi_fit_h_torch = get_torch_regressor_mat(h_fit_torch.view(-1), n_a)
    y_meas_fit_torch = torch.tensor(y_meas_fit)
    u_fit_torch = torch.tensor(u_fit)

    # Setup model an simulator
    io_model = NeuralIOModel(n_a=n_a, n_b=n_b, n_feat=64)
    io_solution = NeuralIOSimulator(io_model)
    #io_solution.io_model.load_state_dict(torch.load(os.path.join("models", "model_IO_1step_nonoise.pkl")))
    params = list(io_solution.io_model.parameters()) #+ [h_fit_torch]
    optimizer = optim.Adam(params, lr=1e-4)
    end = time.time()
    loss_meter = RunningAverageMeter(0.97)


    def get_batch(batch_size, seq_len):
        num_train_samples = y_meas_fit_torch.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch all indices
        batch_idx_seq_h =  batch_start[:, np.newaxis]  - 1 - np.arange(n_a)

        batch_h_seq = h_fit_torch[[batch_idx_seq_h + n_a]].squeeze() # regressor of hidden variables
        batch_u_seq =  torch.tensor(phi_fit_u[batch_start]) # regressor of input variables

        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_h = h_fit_torch[[batch_idx + n_a]]
        return batch_u, batch_y_meas, batch_h, batch_h_seq, batch_u_seq, batch_start

    def get_sequential_batch(seq_len):
        num_train_samples = y_meas_fit_torch.shape[0]
        batch_size = num_train_samples//seq_len
        batch_start = np.arange(0, batch_size, dtype=np.int64) * seq_len
        batch_idx = batch_start[:,np.newaxis] + np.arange(seq_len) # batch all indices
        batch_idx_seq_h =  batch_start[:, np.newaxis]  - 1 - np.arange(n_a)

        batch_h_seq = h_fit_torch[[batch_idx_seq_h + n_a]].squeeze()  # regressor of hidden variables
        batch_u_seq = torch.tensor(phi_fit_u[batch_start])  # regressor of input variables

        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_h = h_fit_torch[[batch_idx + n_a]]
        return batch_u, batch_y_meas, batch_h, batch_h_seq, batch_u_seq, batch_start


    with torch.no_grad():
        batch_u, batch_y_meas, batch_h, batch_h_seq, batch_u_seq, batch_s  = get_batch(batch_size, seq_len)
        batch_y_pred = io_solution.f_sim_minibatch(batch_u, batch_h_seq, batch_u_seq)
        err = batch_y_meas - batch_y_pred
        loss = torch.mean(err ** 2)
        loss_scale = np.float32(loss)

    LOSS = []
    ii = 0
    start_time = time.time()
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Predict
#        batch_u, batch_y_meas, batch_h, batch_h_seq, batch_u_seq, batch_s = get_batch(batch_size, seq_len)
        batch_u, batch_y_meas, batch_h, batch_h_seq, batch_u_seq, batch_start = get_sequential_batch(seq_len)
        batch_y_pred = io_solution.f_sim_minibatch(batch_u, batch_h_seq, batch_u_seq)

        # Compute loss
        err = batch_y_pred - batch_y_meas #batch_h - batch_y_meas
        loss = torch.mean(err**2)
        loss_sc = loss/loss_scale

        # Append to loss vector
        LOSS.append(loss_sc.item())

        # Optimization step
        loss_sc.backward()
        # optimizer.param_groups[0]['params'][-1].grad = 1e3*optimizer.param_groups[0]['params'][-1].grad
        optimizer.step()

        # Print message
        if itr % test_freq == 0:
            print('Iter {:04d} | Loss {:.6f}, Scaled Loss {:.6f}'.format(itr, loss.item(), loss_sc.item()))
            ii += 1
        end = time.time()

    train_time = time.time() - start_time
    if not os.path.exists("models"):
        os.makedirs("models")


    # Build validation data
    n_val = N
    u_val = u[0:n_val]
    y_val = y[0:n_val]
    y_meas_val = y_noise[0:n_val]

    # Neglect initial values
    y_val = y_val[n_max:,:]
    y_meas_val = y_meas_val[n_max:,:]
    u_val = u_val[n_max:, :]

    y_meas_val_torch = torch.tensor(y_meas_val)

    with torch.no_grad():
        y_seq = np.array(np.flip(y_val[0:n_a].ravel()))
        y_seq_torch = torch.tensor(y_seq)

        u_seq = np.array(np.flip(u_val[0:n_b].ravel()))
        u_seq_torch = torch.tensor(u_seq)

        u_torch = torch.tensor(u_val[n_max:,:])
        y_val_sim_torch = io_solution.f_sim(y_seq_torch, u_seq_torch, u_torch)

        err_val = y_val_sim_torch - y_meas_val_torch[n_max:,:]
        loss_val =  torch.mean((err_val)**2)

    if not os.path.exists("fig"):
        os.makedirs("fig")

    # In[Plot]
    y_val_sim = np.array(y_val_sim_torch)
    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(y_val, 'b', label='True')
    ax[0].plot(y_val_sim, 'r',  label='Sim')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(u_val, label='Input')
    ax[1].legend()
    ax[1].grid(True)


    fig, ax = plt.subplots(1,1, figsize=(5,4))
    ax.plot(np.array(LOSS)/LOSS[0])
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")
    fig_name = "RLC_IO_loss_128step_noise.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
