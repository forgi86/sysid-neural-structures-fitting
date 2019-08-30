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

from symbolic_RLC import fxu_ODE, fxu_ODE_mod, A_nominal, B_nominal
from torchid.ssfitter import NeuralStateSpaceSimulator, RunningAverageMeter
from torchid.ssmodels import NeuralStateSpaceModelLin, NeuralStateSpaceModel


if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    t =  np.array(df_X[COL_T], dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])

    Ts = time_data[1] - time_data[0]

    n_x = 2
    n_u = 1
    n_hidden = 64
    ss_model = NeuralStateSpaceModel(n_x, n_u, n_hidden)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", "model_ARX_FE_sat.pkl")))

    x_torch = torch.tensor(x)
    x0_torch = torch.tensor(x[0,:])
    u_torch = torch.tensor(u)
    with torch.no_grad():
        x_sim_torch = nn_solution.f_sim(x0_torch, u_torch)
        loss = torch.mean(torch.abs(x_sim_torch - x_torch))

    x_sim = np.array(x_sim_torch)

    n_plot = t.size
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(t[:n_plot], x[:n_plot, 0], label='True')
    ax[0].plot(t[:n_plot], x_sim[:n_plot, 0], label='Simulated')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Capacitor voltage (V)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t[:n_plot], x[:n_plot, 1], label='True')
    ax[1].plot(t[:n_plot], x_sim[:n_plot, 1], label='Simulated')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Inductor current (A)")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[:n_plot], u[:n_plot, 0])
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Input voltage (V)")
    #ax[2].legend()
    ax[2].grid()

    # In[Kalman filter setup]
    n_x = 2
    n_u = 1
    VAR = []
    for idx_var in range(n_x):
        var = np.zeros((1,n_x)).astype(np.float32)
        var[0,idx_var] = 1.0 # differentiate w.r.t the nth variable
        VAR.append(torch.tensor(var))

    # In[Kalman filter]
    C = np.array([[1., 0.]], dtype=np.float32)

    Q_kal = np.diag([0.1, 1]).astype(np.float32)
    R_kal = 1.0 * np.eye(1).astype(np.float32),

    x_est_post_vec = np.zeros((t.size, n_x)).astype(np.float32)
    x_est_pri_vec = np.zeros((t.size, n_x)).astype(np.float32)

    x_est_pri = x[0, :]  # x[0|-1]
    P_pri = np.eye(n_x, n_x).astype(np.float32)  # P[0|-1]
    I_nx = np.eye(n_x, n_x).astype(np.float32)

    for time_idx in range(len(t)):
        ui = u[time_idx, :]
        yi = y[time_idx, :]

        xi_torch = torch.tensor(x_est_pri, requires_grad=True)  # measurement
        ui_torch = torch.tensor(ui, requires_grad=True)

        x_est_pri_vec[time_idx] = x_est_pri

        f_xu = ss_model(xi_torch, ui_torch)
        Ak = np.empty((n_x, n_x), dtype=np.float32)
        Bk = np.empty((n_x, n_u), dtype=np.float32)
        for idx_var in range(n_x):
            var = VAR[idx_var]
            f_xu.backward(var, retain_graph=True)
            Ak[idx_var, :] = np.array(xi_torch.grad)
            Bk[idx_var, :] = np.array(ui_torch.grad)
            xi_torch.grad.data.zero_()
            ui_torch.grad.data.zero_()
        Ak = Ak + I_nx
        Ck = C

        y_est_pri = Ck @ x_est_pri  # y[k|k-1]
        Sk = Ck @ P_pri @ Ck.T + R_kal  # Innovation covariance
        Kk = P_pri @ Ck.T @ np.linalg.inv(Sk)
        x_est_post = x_est_pri + Kk @ (yi - y_est_pri)  # x[k|k]

        P_post = (I_nx - Kk @ Ck) @ P_pri  # P[k|k]
        x_est_post_vec[time_idx, :] = x_est_post

        f_xu_np = f_xu.clone().data.cpu().numpy()
        x_est_pri = x_est_post + f_xu_np  # x[k+1|k] predict step
        x_est_pri = x_est_pri.ravel()

        P_pri = Ak @ P_post @ Ak.T + Q_kal  # P[k|k-1]


    fig,ax = plt.subplots(3,1, sharex=True)
    ax[0].plot(t[:n_plot], x[:n_plot, 0], label='True')
    ax[0].plot(t[:n_plot], x_est_post_vec[:n_plot, 0], label='Predicted')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Capacitor voltage (V)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t[:n_plot], x[:n_plot, 1], label='True')
    ax[1].plot(t[:n_plot], x_est_post_vec[:n_plot, 1], label='Predicted')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Inductor current (A)")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[:n_plot], u[:n_plot, 0])
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Input voltage (V)")
    #ax[2].legend()
    ax[2].grid()
