import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(".."))
from torchid.neuralode import NeuralODE
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

    # In[Kalman filter setup]
    n_x = 4
    n_u = 1
    VAR = []
    for idx_var in range(n_x):
        var = np.zeros((1,n_x)).astype(np.float32)
        var[0,idx_var] = 1.0 # differentiate w.r.t the nth variable
        VAR.append(torch.tensor(var))


    # In[Kalman filter]
    C = np.array([[1., 0., 0., 0.],
                 [0., 0., 1., 0.]], dtype=np.float32)
    
    Q_kal = np.diag([0.1, 10, 0.1, 10]).astype(np.float32)
    R_kal = 1.0*np.eye(2).astype(np.float32),
    
    x_est_post_vec = np.zeros((t.size, n_x)).astype(np.float32)
    x_est_pri_vec  = np.zeros((t.size, n_x)).astype(np.float32)

    x_est_pri = x[0, :] # x[0|-1]
    P_pri = np.eye(n_x, n_x).astype(np.float32) # P[0|-1]
    I_nx = np.eye(n_x, n_x).astype(np.float32)

    for time_idx in range(len(t)):
        ui = u[time_idx,:]
        yi = y[time_idx,:]

        xi_torch = torch.tensor(x_est_pri, requires_grad=True) # measurement
        ui_torch = torch.tensor(ui, requires_grad=True)

        x_est_pri_vec[time_idx] = x_est_pri

        f_xu = ss_model(xi_torch, ui_torch)
        Ak = np.empty((n_x, n_x),dtype=np.float32)
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

        y_est_pri = Ck @ x_est_pri # y[k|k-1]
        Sk = Ck @ P_pri @ Ck.T + R_kal # Innovation covariance
        Kk = P_pri @ Ck.T @ np.linalg.inv(Sk)
        x_est_post = x_est_pri + Kk @ (yi - y_est_pri) # x[k|k]

        P_post = (I_nx - Kk @ Ck) @ P_pri # P[k|k]
        x_est_post_vec[time_idx,:] = x_est_post

        f_xu_np = f_xu.clone().data.cpu().numpy()
        x_est_pri = x_est_post + f_xu_np   # x[k+1|k] predict step
        x_est_pri = x_est_pri.ravel()

        P_pri = Ak @ P_post @ Ak.T + Q_kal # P[k|k-1]



    fig,ax = plt.subplots(4,1,figsize=(20,10), sharex=True)
    ax[0].plot(t[:n_plot], x[:n_plot, 0], label='True')
    ax[0].plot(t[:n_plot], x_est_post_vec[:n_plot, 0], label='Predicted')
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position p (m)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t[:n_plot], x[:n_plot, 1], label='True')
    ax[1].plot(t[:n_plot], x_est_post_vec[:n_plot, 1], label='Predicted')
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Speed v (m/s)")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[:n_plot], x[:n_plot, 2], label='True')
    ax[2].plot(t[:n_plot], x_est_post_vec[:n_plot, 2], label='Predicted')
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle $\phi$ (rad)")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t[:n_plot], x[:n_plot, 3], label='True')
    ax[3].plot(t[:n_plot], x_est_post_vec[:n_plot, 3], label='Predicted')
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Angular velocity $\omega$ (rad/s)")
    ax[3].legend()
    ax[3].grid()
