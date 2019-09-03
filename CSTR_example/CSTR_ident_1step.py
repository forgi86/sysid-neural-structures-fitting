import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(".."))
from torchid.ssfitter import  NeuralStateSpaceSimulator
from torchid.util import RunningAverageMeter
from torchid.ssmodels import NeuralStateSpaceModel


if __name__ == '__main__':

    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    df_X = pd.read_csv(os.path.join("data", "cstr.dat"), header=None, sep="\t")
    df_X.columns = ['time', 'q', 'Ca', 'T', 'None']

    df_X['q'] = df_X['q']/100
    df_X['Ca'] = df_X['Ca']*10
    df_X['T'] = df_X['T']/400
    
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])


    x_noise = np.copy(x) #+ np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)

    Ts = time_data[1] - time_data[0]
    t_fit = time_data[-1]
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 20000
    test_freq = 100

    input_data = u[0:n_fit]
    state_data = x_noise[0:n_fit]
    u_torch = torch.from_numpy(input_data)
    x_true_torch = torch.from_numpy(state_data)
    
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    #nn_solution.load_state_dict(torch.load(os.path.join("models", "model_ARX_FE_sat.pkl")))

    optimizer = optim.Adam(nn_solution.ss_model.parameters(), lr=1e-4)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    
#    scale_error = 1./np.std(x_noise, axis=0)
#    scale_error = scale_error/np.sum(scale_error)
    scale_error = torch.tensor(1e2)

    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()
        x_pred_torch = nn_solution.f_onestep(x_true_torch, u_torch)
        err = x_pred_torch - x_true_torch
        err_scaled = err * scale_error
        loss = torch.mean((err_scaled)**2) #torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % test_freq == 0:
#            with torch.no_grad():
#                x_pred_torch = nn_solution.f_onestep(x_true_torch, u_torch) #func(x_true_torch, u_torch)
#                loss = torch.mean((x_pred_torch - x_true_torch) ** 2)
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            ii += 1
        end = time.time()

    if not os.path.exists("models"):
        os.makedirs("models")
    
    torch.save(nn_solution.ss_model.state_dict(), os.path.join("models", "model_ARX_FE_sat_nonoise.pkl"))

    x_0 = state_data[0,:]

    time_start = time.time()
    with torch.no_grad():
        x_sim = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(input_data))
        loss = torch.mean(torch.abs(x_sim - x_true_torch))
    time_arr = time.time() - time_start


    # In[Plot]
    x_sim = np.array(x_sim)
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(np.array(x_true_torch[:,0]), 'k+',  label='True')
    ax[0].plot(np.array(x_pred_torch[:,0].detach()), 'b', label='Pred')
    ax[0].plot(x_sim[:,0],'r', label='Sim')
    ax[0].legend()
    ax[1].plot(np.array(x_true_torch[:,1]), 'k+', label='True')
    ax[1].plot(np.array(x_pred_torch[:,1].detach()), 'b', label='Pred')
    ax[1].plot(x_sim[:,1],'r', label='Sim')
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
