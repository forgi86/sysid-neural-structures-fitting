import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(".."))
from torchid.ssfitter_jit import  NeuralStateSpaceSimulator
from torchid.util import RunningAverageMeter
from torchid.ssmodels import NeuralStateSpaceModel

if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])

    std_noise_V = 0.0 * 10.0
    std_noise_I = 0.0 * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)

    Ts = time_data[1] - time_data[0]
    t_fit = 0.5e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 1000
    test_freq = 100

    input_data = u[0:n_fit]
    state_data = x_noise[0:n_fit]
    u_torch = torch.from_numpy(input_data)
    x_true_torch = torch.from_numpy(state_data)

    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    params = list(nn_solution.ss_model.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    end = time.time()

    func = torch.jit.trace(nn_solution, (x0_torch, u_torch))

    with torch.no_grad():
        x_est_torch = func(x0_torch, u_torch) #nn_solution.f_sim
        err_init = x_est_torch  - x_true_torch
        scale_error = torch.sqrt(torch.mean((err_init)**2, dim=(0))) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))

    start_time = time.time()
    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()
        x_est_torch = func(x0_torch, u_torch)#nn_solution.f_sim(x0_torch, u_torch)
        err = x_est_torch - x_true_torch
        err_scaled = err/scale_error
        loss = torch.mean(err_scaled ** 2)

        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            ii += 1

        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time

#    torch.save(nn_solution.state_dict(), 'model.pkl')

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
