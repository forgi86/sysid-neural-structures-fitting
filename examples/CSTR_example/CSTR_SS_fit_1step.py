import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join("..", ".."))
from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel

if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Overall paramaters
    num_iter = 10000
    lr=1e-4
    test_freq = 100
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
    x0_torch = torch.from_numpy(x[0,:])
    x_noise = np.copy(x) # np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    Ts = time_data[1] - time_data[0]

    # Get fit data
    t_fit = time_data[-1] # use all data
    n_fit = int(t_fit//Ts)
    input_data = u[0:n_fit]
    state_data = x_noise[0:n_fit]
    u_torch = torch.from_numpy(input_data)
    x_true_torch = torch.from_numpy(state_data)

    # Setup neural model structure
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64, init_small=False)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    # Setup optimizer
    optimizer = optim.Adam(nn_solution.ss_model.parameters(), lr=lr)

    # Scale loss with respect to the initial one
    with torch.no_grad():
        x_est_torch = nn_solution.f_onestep(x_true_torch, u_torch)
        err_init = x_est_torch - x_true_torch
        scale_error = torch.sqrt(torch.mean(err_init**2, dim=0))

    # Training loop
    LOSS = []
    start_time = time.time()
    for itr in range(0, num_iter):

        optimizer.zero_grad()
        x_est_torch = nn_solution.f_onestep(x_true_torch, u_torch)
        err = x_est_torch - x_true_torch
        err_scaled = err / scale_error
        loss_sc = torch.mean((err_scaled) ** 2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))

        if itr % test_freq == 0:
            with torch.no_grad():
                loss_unsc = torch.mean(err**2)
                print('Iter {:04d} | Loss {:.6f}, Scaled Loss {:.6f}'.format(itr, loss_unsc.item(), loss_sc.item()))

        LOSS.append(loss_sc.item())
        loss_sc.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    if not os.path.exists("models"):
        os.makedirs("models")
    
    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", "model_SS_1step_nonoise.pkl"))


    # In[Plot]
    x_0 = state_data[0,:]
    time_start = time.time()
    with torch.no_grad():
        x_sim = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(input_data))
        loss_sc = torch.mean(torch.abs(x_sim - x_true_torch))
    time_arr = time.time() - time_start

    x_sim = np.array(x_sim)
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(np.array(x_true_torch[:,0]), 'k',  label='True')
    ax[0].plot(x_sim[:,0],'r', label='Sim')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(np.array(x_true_torch[:,1]), 'k', label='True')
    ax[1].plot(x_sim[:,1],'r', label='Sim')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(np.array(u_torch), 'b', label='Input')
    ax[2].legend()
    ax[2].grid(True)

    if not os.path.exists("fig"):
        os.makedirs("fig")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6))
    ax.plot(LOSS)
    ax.grid(True)
    ax.set_ylabel("Loss (-)")
    ax.set_xlabel("Iteration (-)")

    if add_noise:
        fig_name = "CSTR_SS_loss_1step_noise.pdf"
    else:
        fig_name = "CSTR_SS_loss_1step_nonoise.pdf"

    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')
