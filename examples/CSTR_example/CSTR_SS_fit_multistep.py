import os
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt


from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel


if __name__ == '__main__':

    num_iter = 5000
    test_freq = 100
    seq_len = 128
    lr = 1e-3
    add_noise = False

    # Column names
    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "CSTR_data_id.csv"))
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = time_data[1] - time_data[0]

    # Add noise
    x_noise = np.copy(x) #+ np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)

    # Get fit data
    t_fit = time_data[-1]
    n_fit = int(t_fit//Ts)#x.shape[0]
    u_fit = u[0:n_fit]
    x_fit = x_noise[0:n_fit]
    y_fit = y[0:n_fit]
    time_fit = time_data[0:n_fit]
    batch_size = n_fit//seq_len

    # Fit data to pytorch tensors #
    u_torch_fit = torch.from_numpy(u_fit)
    y_true_torch_fit = torch.from_numpy(y_fit)
    x_meas_torch_fit = torch.from_numpy(x_fit)
    time_torch_fit = torch.from_numpy(time_fit)

    def get_batch(batch_size, seq_len):
        num_train_samples = x_meas_torch_fit.shape[0]
        batch_start = np.random.choice(np.arange(num_train_samples - seq_len, dtype=np.int64), batch_size, replace=False)
        batch_idx = batch_start[:, np.newaxis] + np.arange(seq_len) # batch all indices

        batch_t = torch.tensor(time_fit[batch_idx]) #torch.stack([time_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x0 = torch.tensor(x_fit[batch_start]) #x_meas_torch_fit[batch_start, :]  # (M, D)
        batch_u = torch.tensor(u_fit[batch_idx]) #torch.stack([u_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)
        batch_x = torch.tensor(x_fit[batch_idx]) #torch.stack([x_meas_torch_fit[batch_start[i]:batch_start[i] + seq_len] for i in range(batch_size)], dim=0)

        return batch_t, batch_x0, batch_u, batch_x 

    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64) #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    #nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_SS_1step_nonoise.pkl")))

    # Setup optimizer
    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=lr)
    end = time.time()

    with torch.no_grad():
        batch_t, batch_x0, batch_u, batch_x = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution.f_sim_multistep(batch_x0, batch_u)
        err_init = batch_x_sim - batch_x
        scale_error = torch.sqrt(torch.mean((err_init)**2,dim=(0,1)))

    
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()
        batch_t, batch_x0, batch_u, batch_x = get_batch(batch_size, seq_len)
        batch_x_sim = nn_solution.f_sim_multistep(batch_x0, batch_u)
        err = batch_x_sim - batch_x
        err_scaled = err/scale_error
        loss_sc = torch.mean(err_scaled**2)

        if itr % test_freq == 0:
            with torch.no_grad():
                loss_unsc = torch.mean(err**2)
                print('Iter {:04d} | Loss {:.6f}, Scaled Loss {:.6f}'.format(itr, loss_unsc.item(), loss_sc.item()))

        LOSS.append(loss_sc.item())
        loss_sc.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    if add_noise:
        model_filename = f"model_SS_{seq_len}step_noise_from1step.pkl"
    else:
        model_filename = f"model_SS_{seq_len}step_nonoise_from1step.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))

    # In[Validate model]
    t_val = time_data[-1]
    n_val = int(t_val//Ts)#x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]
    output_data_val = y[0:n_val]

    x0_val = state_data_val[0,:]#np.zeros(2,dtype=np.float32)
    x0_torch_val = torch.from_numpy(x0_val)
    u_torch_val = torch.tensor(input_data_val)
    x_true_torch_val = torch.from_numpy(state_data_val)

    with torch.no_grad():
        x_pred_torch_val = nn_solution.f_sim(x0_torch_val, u_torch_val)

    # In[1]

    fig,ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(np.array(x_true_torch_val[:,0]), label='True')
    ax[0].plot(np.array(x_pred_torch_val[:,0]), label='Fit')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_true_torch_val[:,1]), label='True')
    ax[1].plot(np.array(x_pred_torch_val[:,1]), label='Fit')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch_val), label='Input')
    ax[2].grid(True)


    fig, ax = plt.subplots(1,1)
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = f"CSRT_SS_loss_{seq_len}step_noise.pdf"
    else:
        fig_name = f"CSTR_SS_loss_{seq_len}step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
