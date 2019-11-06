import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join("..", '..'))
from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel

if __name__ == '__main__':

    # Overall paramaters
    t_fit = 2e-3 #2e-3
    num_iter = 10000
    test_freq = 10
    add_noise = True

    # Column names in the dataset
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])

    # Add measurement noise
    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])
    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)

    Ts = time_data[1] - time_data[0]
    n_fit = int(t_fit//Ts)#x.shape[0]

    # Fit data to pytorch tensors #
    input_data = u[0:n_fit]
    state_data = x_noise[0:n_fit]
    u_torch = torch.from_numpy(input_data)
    x_true_torch = torch.from_numpy(state_data)

    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    # Setup optimizer
    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    # Scale loss with respect to the initial one
    with torch.no_grad():
        x_est_torch = nn_solution.f_sim(x0_torch, u_torch)
        err_init = x_est_torch  - x_true_torch
        scale_error = torch.sqrt(torch.mean((err_init)**2, dim=(0)))

    start_time = time.time()
    LOSS = []
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()

        # Perform open-loop simulation
        x_est_torch = nn_solution.f_sim(x0_torch, u_torch)

        # Compute fit loss
        err = x_est_torch - x_true_torch
        err_scaled = err/scale_error
        loss = torch.mean(err_scaled ** 2)

        # Statistics
        LOSS.append(loss.item())
        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

        # Optimize
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}") # 7312.95 seconds!
    
    if not os.path.exists("models"):
        os.makedirs("models")

    if add_noise:
        model_filename = "model_SS_simerr_noise.pkl"
    else:
        model_filename = "model_SS_simerr_nonoise.pkl"

    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", model_filename))

    t_val = 5e-3
    n_val = int(t_val//Ts)#x.shape[0]

    input_data_val = u[0:n_val]
    state_data_val = x[0:n_val]
    output_data_val = y[0:n_val]

    x0_val = np.zeros(2,dtype=np.float32)
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

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig,ax = plt.subplots(1,1, figsize=(7.5,6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = "RLC_SS_loss_simerr_noise.pdf"
    else:
        fig_name = "RLC_SS_loss_simerr_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
