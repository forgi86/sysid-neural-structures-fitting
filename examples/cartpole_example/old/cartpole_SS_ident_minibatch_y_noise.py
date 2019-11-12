import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join("..", ".."))
from torchid.ssfitter import NeuralStateSpaceSimulator
from torchid.ssmodels import CartPoleStateSpaceModel
import scipy.signal as signal


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall parameters
    len_fit = 80
    seq_len = 32
    test_freq = 50
    num_iter = 100000
    test_freq = 50
    add_noise = True
    lr = 1e-4  # try 4e-6

    std_noise_p = add_noise * 0.02
    std_noise_theta = add_noise * 0.004
    std_noise = np.array([std_noise_p, std_noise_theta])

    # Column names in the dataset
    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_oloop_id.csv"))

    # Load dataset
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    # Add measurement noise
    y_meas = np.copy(y) + np.random.randn(*y.shape) * std_noise

    # Design a differentiator filter to estimate unmeasured velocities from noisy, measured positions
    Ts = np.float(t[1] - t[0])
    fs = 1/Ts       # Sample rate, Hz
    cutoff = 1.0    # Desired cutoff frequency, Hz
    trans_width = 5  # Width of transition from pass band to stop band, Hz
    numtaps = 64      # Size of the FIR filter.
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [2*np.pi*2*np.pi*10*1.5, 0], Hz=fs, type='differentiator')

    # Filter positions to estimate velocities
    n_x = x.shape[-1] # 4
    x_est = np.zeros((y_meas.shape[0], 4), dtype=np.float32)
    x_est[:, 0] = y_meas[:, 0]
    x_est[:, 2] = y_meas[:, 1]
    x_est[:, 1] = np.convolve(x_est[:, 0], taps, 'same')
    x_est[:, 3] = np.convolve(x_est[:, 2], taps, 'same')

    # Get fit data
    n_fit = int(len_fit//Ts)
    time_fit = t[0:n_fit]
    u_fit = u[0:n_fit]
    x_est_fit = x_est[0:n_fit]
    y_meas_fit = y[0:n_fit]
    batch_size = n_fit//seq_len
    x_hidden_fit = torch.tensor(x_est_fit, requires_grad=True)  # this is an optimization variable!


    # Setup neural model structure
    ss_model = CartPoleStateSpaceModel(Ts, init_small=True)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    # Setup optimizer
    params = list(nn_solution.ss_model.parameters()) + [x_hidden_fit]
    optimizer = optim.Adam(params, lr=lr)

    # Batch extraction funtion
    def get_sequential_batch(seq_len):

        # Select batch indexes
        num_train_samples = x_est_fit.shape[0]
        batch_size = num_train_samples//seq_len-1
        batch_start = np.arange(0, batch_size, dtype=np.int64) * seq_len
        batch_idx = batch_start[:,np.newaxis] + np.arange(seq_len) # batch all indices

        # Extract batch data
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])

        return batch_t, batch_x0_hidden, batch_u, batch_y_meas


    # Scale loss with respect to the initial one
    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_y_meas = get_sequential_batch(seq_len)
        batch_x_sim = nn_solution.f_sim_multistep(batch_x0_hidden, batch_u)
        err_init = batch_x_sim[:, :, [0, 2]] - batch_y_meas
        scale_error = torch.sqrt(torch.mean(err_init**2,dim=(0,1)))
        
    LOSS = []
    start_time = time.time()
    # Training loop
    for itr in range(0, num_iter):
        optimizer.zero_grad()

        # Simulate
        batch_t, batch_x0_hidden, batch_u, batch_y_meas = get_sequential_batch(seq_len)
        batch_x_sim = nn_solution.f_sim_multistep(batch_x0_hidden, batch_u)

        # Compute fit loss
        err = batch_x_sim[:, :, [0, 2]] - batch_y_meas
        err_scaled = err/scale_error
        loss = torch.mean(err_scaled**2)

        # Statistics
        LOSS.append(loss.item())
        if itr > 0 and itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # Save model
    if not os.path.exists("models"):
        os.makedirs("models")
    if add_noise:
        model_filename = f"model_SS_{seq_len}step_noise.pkl"
    else:
        model_filename = f"model_SS_{seq_len}step_nonoise.pkl"
    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))

    # Simulate model
    x_0 = x_est_fit[0, :]
    with torch.no_grad():
        x_meas_fit_torch = torch.tensor(x_est_fit)
        x_sim_torch = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(u_fit))
        loss = torch.mean(torch.abs(x_sim_torch - x_meas_fit_torch))
        x_sim = np.array(x_sim_torch)

    n_plot = 100
    idx_plot_start = 0

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(time_fit[idx_plot_start:idx_plot_start+n_plot,:], x_est_fit[idx_plot_start:idx_plot_start + n_plot, 0], label='True')
    ax[0].plot(time_fit[idx_plot_start:idx_plot_start+n_plot], x_sim[idx_plot_start:idx_plot_start+n_plot, 0], label='Simulated')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (m)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(time_fit[idx_plot_start:idx_plot_start+n_plot], x_est_fit[idx_plot_start:idx_plot_start+n_plot, 2], label='True')
    ax[1].plot(time_fit[idx_plot_start:idx_plot_start+n_plot], x_sim[idx_plot_start:idx_plot_start+n_plot, 2], label='Simulated')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle (rad)")
    ax[1].legend()
    ax[1].grid()

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = f"cartpole_SS_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"cartpole_SS_loss_{seq_len}step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
