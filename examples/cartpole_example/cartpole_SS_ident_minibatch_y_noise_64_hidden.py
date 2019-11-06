import os
import time
import numpy as np
import sys
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(".."))
from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.ssmodels import CartPoleStateSpaceModel
import scipy.signal as signal
from tensorboardX import SummaryWriter



# In[Load data]
if __name__ == '__main__':

    len_fit = 80
    seq_len = 64 # try 32
    test_freq = 50
    num_iter = 200000
    test_freq = 50
    add_noise = True
    lr = 2e-5 # try 4e-6

    std_noise_p = add_noise * 0.02
    std_noise_theta = add_noise * 0.004
    std_noise = np.array([std_noise_p, std_noise_theta])

    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
#    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC_ref_id.csv"))
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_oloop_id.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    y_meas = np.copy(y) + np.random.randn(*y.shape) * std_noise
    Ts = np.float(t[1] - t[0])

    Ts = np.float(t[1] - t[0])
    fs = 1/Ts       # Sample rate, Hz
    cutoff = 1.0    # Desired cutoff frequency, Hz
    trans_width = 5  # Width of transition from pass band to stop band, Hz
    numtaps = 64      # Size of the FIR filter.
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [2*np.pi*2*np.pi*10*1.5, 0], Hz=fs, type='differentiator')


    #x_est = x
    x_est = np.zeros((y_meas.shape[0], 4), dtype=np.float32)
    x_est[:, 0] = y_meas[:, 0]
    x_est[:, 2] = y_meas[:, 1]
    x_est[:, 1] = np.convolve(x_est[:, 0], taps, 'same')  # signal.lfilter(taps, 1, y_meas[:,0])*2*np.pi
    x_est[:, 3] = np.convolve(x_est[:, 2], taps, 'same')  # signal.lfilter(taps, 1, y_meas[:,1])*2*np.pi

    n_x = x.shape[-1]
    ss_model = CartPoleStateSpaceModel(Ts, init_small=True)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    #model_name = "model_SS_64step_noise.pkl"
    #model_name = "model_SS_150step_nonoise.pkl"
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_name )))

    n_fit = int(len_fit//Ts)
    time_fit = t[0:n_fit]
    u_fit = u[0:n_fit]
    x_est_fit = x_est[0:n_fit]
    y_meas_fit = y[0:n_fit]
    batch_size = n_fit//seq_len
    x_hidden_fit = torch.tensor(x_est_fit, requires_grad=True)  # this is an optimization variable!
    x_fit_ideal = x[0:n_fit] # not to be used, just for plots


    params = list(nn_solution.ss_model.parameters()) + [x_hidden_fit]
    optimizer = optim.Adam(params, lr=lr)
    end = time.time()


    def get_sequential_batch(seq_len):
        num_train_samples = x_est_fit.shape[0]
        batch_size = num_train_samples//seq_len-1
        batch_start = np.arange(0, batch_size, dtype=np.int64) * seq_len
        batch_idx = batch_start[:,np.newaxis] + np.arange(seq_len) # batch all indices

        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])
        return batch_t, batch_x0_hidden, batch_u, batch_y_meas

    def get_random_batch(batch_size, seq_len):
        num_train_samples = x_est_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch all indices

        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[batch_idx]
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_y_meas = torch.tensor(y_meas_fit[batch_idx])
        batch_u = torch.tensor(u_fit[batch_idx])

        return batch_t, batch_x0_hidden, batch_u, batch_y_meas, batch_x_hidden

    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_y_meas, batch_x_hidden = get_random_batch(batch_size, seq_len)
        batch_x_sim = nn_solution.f_sim_multistep(batch_x0_hidden, batch_u)
        err_init = batch_x_sim - batch_x_hidden
        scale_error = torch.sqrt(torch.mean((err_init)**2,dim=(0,1)))
        scale_error[2] = scale_error[2]
        
# In[Fit model]
    LOSS = []

#    writer = SummaryWriter(comment="-" + args.env)
    writer = SummaryWriter("logs")

    for itr in range(0, num_iter):
        optimizer.zero_grad()

        batch_t, batch_x0_hidden, batch_u, batch_y_meas, batch_x_hidden = get_random_batch(batch_size, seq_len)
        batch_x_sim = nn_solution.f_sim_multistep(batch_x0_hidden, batch_u)

        err_consistency = batch_x_hidden  - batch_x_sim
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        err_fit = batch_x_sim[:, :, [0, 2]] - batch_y_meas
        err_fit_scaled = err_fit/scale_error[[0,2]]
        loss_fit = torch.mean(err_fit_scaled**2)

        loss = 1.0*loss_consistency + 1.0*loss_fit

        writer.add_scalar("loss", loss, itr)
        writer.add_scalar("loss_consistency", loss_consistency, itr)
        writer.add_scalar("loss_fit", loss_fit, itr)

        LOSS.append(loss.item())

        if itr > 0 and itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Total Loss {loss:.6f}   Consistency Loss {loss_consistency:.8f}   Fit Loss {loss_fit:.8f}')

        loss.backward()
        optimizer.step()

        end = time.time()

    # In[Save model parameters]
    if not os.path.exists("models"):
        os.makedirs("models")

    if add_noise:
        model_filename = f"model_SS_{seq_len}step_noise_hidden.pkl"
    else:
        model_filename = f"model_SS_{seq_len}step_nonoise_hidden.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))

# In[Simulate model]
    x_0 = x_est_fit[0, :]
    with torch.no_grad():
        x_meas_fit_torch = torch.tensor(x_est_fit)
        x_sim_torch = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(u_fit))
        loss = torch.mean(torch.abs(x_sim_torch - x_meas_fit_torch))
        x_sim = np.array(x_sim_torch)
    # In[1]

    n_plot = 100
    idx_plot_start = 1000

    fig,ax = plt.subplots(2,1,sharex=True)
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




    x_est_optimized = np.array(x_hidden_fit.detach())
    x_est_orig = np.copy(x_est_fit)
    x_est_optimized_diff = np.copy(x_est_optimized)
    x_est_optimized_diff[x_est_optimized == x_est_orig] = np.nan

    fig, ax = plt.subplots(4,1, figsize=(7.5,6))
    for idx, a in enumerate(ax):
        a.plot(x_est_orig[:,idx], '-k')
        a.plot(x_est_optimized_diff[:,idx], '-r')
        a.plot(x_fit_ideal[:,idx], '-g')


    fig, ax = plt.subplots(4,1, figsize=(7.5,6))
    for idx, a in enumerate(ax):
        a.plot(x_est_optimized[:, idx] - x_est_orig[:, idx], '*')
