import os
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(".."))
from torchid.neuralode import NeuralODE, RunningAverageMeter
from torchid.ssmodels import MechanicalStateSpaceModel


if __name__ == '__main__':

    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = t[1] - t[0]

    x0_torch = torch.from_numpy(x[0,:])

    ss_model = MechanicalStateSpaceModel(Ts)
    nn_solution = NeuralODE(ss_model)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_OE_minibatch.pkl")))

    x_torch = torch.tensor(x)
    x0_torch = torch.tensor(x[0,:])
    u_torch = torch.tensor(u)
    with torch.no_grad():
        x_sim_torch = nn_solution.f_OE(x0_torch, u_torch)
        loss = torch.mean(torch.abs(x_sim_torch - x_torch))

    x_sim = np.array(x_sim_torch)

    n_plot = t.size
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(t[:n_plot], x[:n_plot, 0], label='True')
    ax[0].plot(t[:n_plot], x_sim[:n_plot, 0], label='Simulated')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Cart position (m)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t[:n_plot], x[:n_plot, 2], label='True')
    ax[1].plot(t[:n_plot], x_sim[:n_plot, 2], label='Simulated')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Pendulum angle (rad)")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[:n_plot], u[:n_plot, 0])
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Input Voltage (V)")
    #ax[2].legend()
    ax[2].grid()
"""
    VAR = []
    for idx_var in range(n_x):
        var = np.zeros((1,n_x)).astype(np.float32)
        var[0,idx_var] = 1.0 # differentiate w.r.t the nth variable
        VAR.append(torch.tensor(var))

    F_xu = ss_model(x_torch,u_torch)
    A = np.empty((n_x,n_x))
    B = np.empty((n_x,n_u))
        
    for idx_var in range(n_x):
        var = VAR[idx_var]
        F_xu.backward(var, retain_graph=True)
        A[idx_var,:] = np.array(x_torch.grad)
        B[idx_var,:] = np.array(u_torch.grad)
        x_torch.grad.data.zero_()
        u_torch.grad.data.zero_()
"""
