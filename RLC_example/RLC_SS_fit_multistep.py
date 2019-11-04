import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt

from torchid.ssfitter import NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel

if __name__ == '__main__':

    np.random.seed(42)
    num_iter = 20000  #5000  # 10000
    seq_len = 64  # int(n_fit/10)
    test_freq = 100
    t_fit = 2e-3
    alpha = 0.5
    lr = 2e-4
    add_noise = True

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    x0_torch = torch.from_numpy(x[0, :])

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)

    Ts = t[1] - t[0]
    n_fit = int(t_fit // Ts)  # x.shape[0]
    batch_size = n_fit // seq_len

    # Get fit data #
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    x_fit_nonoise = x[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = t[0:n_fit]

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    y_true_torch_fit = torch.from_numpy(y_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    x_hidden_fit = torch.tensor(x_fit, requires_grad=True)  # this is an optimization variable!

    def get_batch(batch_size, seq_len):

        # Define batch indexes
        num_train_samples = x_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False) # batch start indices
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch samples indices

        # Extract batch data
        batch_t = torch.tensor(time_fit[batch_idx])
        batch_x0_hidden = x_hidden_fit[batch_start, :]
        batch_x_hidden = x_hidden_fit[[batch_idx]]
        batch_u = torch.tensor(u_fit[batch_idx])
        batch_x = torch.tensor(x_fit[batch_idx])

        return batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden


    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    params = list(nn_solution.ss_model.parameters()) + [x_hidden_fit]
    optimizer = optim.Adam(params, lr=lr)

    with torch.no_grad():
        batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution.f_sim_minibatch(batch_x0_hidden, batch_u)
        err_init = batch_x_sim - batch_x
        scale_error = torch.sqrt(torch.mean((err_init) ** 2, dim=(0, 1)))

    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()
        batch_t, batch_x0_hidden, batch_u, batch_x, batch_x_hidden = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution.f_sim_minibatch(batch_x0_hidden, batch_u)

        # Fit loss
        err_fit = batch_x_sim - batch_x
        err_fit_scaled = err_fit/scale_error
        loss_fit = torch.mean(err_fit_scaled**2)

        # Consistency loss
        err_consistency = batch_x_sim - batch_x_hidden
        err_consistency_scaled = err_consistency/scale_error
        loss_consistency = torch.mean(err_consistency_scaled**2)

        loss = alpha*loss_fit + (1.0-alpha)*loss_consistency

        if itr > 0 and itr % test_freq == 0:
            with torch.no_grad():
                print(f'Iter {itr} | Tradeoff Loss {loss:.4f}   Consistency Loss {loss_consistency:.4f}   Fit Loss {loss_fit:.4f}')

        loss.backward()
        optimizer.step()

        LOSS.append(loss.item())

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    if add_noise:
        model_filename = f"model_SS_{seq_len}step_noise.pkl"
    else:
        model_filename = f"model_SS_{seq_len}step_nonoise.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))

    t_val = 5e-3
    n_val = int(t_val // Ts)  # x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]
    output_data_val = y[0:n_val]

    x0_val = np.zeros(2, dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)

    with torch.no_grad():
        x_pred_torch_val = nn_solution.f_sim(x0_torch_val, u_torch_val)

    # In[1]

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.array(x_true_torch_val[:, 0]), label='True')
    ax[0].plot(np.array(x_pred_torch_val[:, 0]), label='Fit')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_true_torch_val[:, 1]), label='True')
    ax[1].plot(np.array(x_pred_torch_val[:, 1]), label='Fit')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_val), label='Input')
    ax[2].grid(True)

    fig, ax = plt.subplots(1, 1)
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = f"RLC_SS_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"RLC_SS_loss_{seq_len}step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')


    x_hidden_fit_np = x_hidden_fit.detach().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_fit_nonoise[:, 0], 'k', label='True')
    ax[0].plot(x_fit[:, 0], 'b', label='Measured')
    ax[0].plot(x_hidden_fit_np[:, 0], 'r', label='Hidden')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(x_fit_nonoise[:, 1], 'k', label='True')
    ax[1].plot(x_fit[:, 1], 'b', label='Measured')
    ax[1].plot(x_hidden_fit_np[:, 1], 'r', label='Hidden')
    ax[1].legend()
    ax[1].grid(True)
