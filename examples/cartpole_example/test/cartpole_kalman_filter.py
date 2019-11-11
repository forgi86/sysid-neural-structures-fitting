import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(".."))
from torchid.ssfitter import NeuralStateSpaceSimulator
from torchid.ssmodels import CartPoleStateSpaceModel


if __name__ == '__main__':

    plt.close("all")
    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC.csv"))

    std_noise_p = 0.01
    std_noise_phi = 0.002
    std_noise = np.array([std_noise_p, std_noise_phi])

    t = np.array(df_X[COL_T], dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    y = np.copy(x[:, [0, 2]])
    u = np.array(df_X[COL_U],dtype=np.float32)
    Ts = t[1] - t[0]
    n_x = x.shape[-1]

    x0_torch = torch.from_numpy(x[0,:])
#    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
#    x_noise = x_noise.astype(np.float32)
    y_noise = np.copy(y) + np.random.randn(*y.shape)*std_noise
    y_noise = y_noise.astype(np.float32)

    # In[Load model]    
    ss_model = CartPoleStateSpaceModel(Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)
    #model_name = "model_OE_minibatch_100.pkl" 
    model_name = "model_ARX_FE_nonoise.pkl"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_name)))
    
    # In[Simulation plot]
    
    x_torch = torch.tensor(x)
    x0_torch = torch.tensor(x[0,:])
    u_torch = torch.tensor(u)
    t_torch = torch.tensor(t)
    with torch.no_grad():
        x_sim_torch = nn_solution.f_sim(x0_torch, u_torch)
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
    ax[2].set_ylabel("Input Force (V)")
    #ax[2].legend()
    ax[2].grid()

    
    # In[Generate batches]
    len_sim = x.shape[0]
    seq_len = 100
    dist_sim = 100
    
    s = np.arange(0, len_sim - seq_len ,dist_sim, dtype = np.int )
    batch_size = len(s)
    batch_x0 = x_torch[s, :]  # (M, D)
    batch_t = torch.stack([t_torch[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_x = torch.stack([x_torch[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)
    batch_u = torch.stack([u_torch[s[i]:s[i] + seq_len] for i in range(batch_size)], dim=0)

    # In[ZOH baseline performance]
    #zoh_error = batch_x -batch_x0.view(batch_size,1,n_x)
    #scale_error = torch.sqrt(torch.mean(zoh_error**2,(0,1)))     

    # In[Predictor performance]

    batch_x_pred = nn_solution.f_sim_multistep(batch_x0, batch_u)
    batch_x_np = batch_x_pred.clone().data.cpu().numpy()
    batch_t_np = batch_t.clone().data.cpu().numpy()
    #err = batch_x[:,1:,:] - batch_x_pred[:,1:,:]
    #err_scaled = err * scale_error        
    #loss = torch.mean(err_scaled**2)
    
    # In[Performance plot]
    
    
    fig,ax = plt.subplots(4,1,figsize=(20,10), sharex=True)
    ax[0].plot(t[:n_plot], y_noise[:n_plot, 0], 'k', label='Measured')
    ax[0].plot(batch_t_np[:,:,0].T, batch_x_np[:,:,0].T, 'r', linewidth=3)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position p (m)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t[:n_plot], x[:n_plot, 1], label='True')
    ax[1].plot(batch_t_np[:,:,0].T, batch_x_np[:,:,1].T, 'r',linewidth=3)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Speed v (m/s)")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t[:n_plot], y_noise[:n_plot, 1], 'k', label='Measured')
    ax[2].plot(batch_t_np[:,:,0].T, batch_x_np[:,:,2].T, 'r',linewidth=3)
    ax[2].plot(t[:n_plot], x[:n_plot, 2], label='True')
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Angle $\phi$ (rad)")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t[:n_plot], x[:n_plot, 3], label='True')
    ax[3].plot(batch_t_np[:,:,0].T, batch_x_np[:,:,3].T, 'r',linewidth=3)
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Angular velocity $\omega$ (rad/s)")
    ax[3].legend()
    ax[3].grid()
    
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
    
    Q_kal = np.diag([0.01, 1, 0.01, 1]).astype(np.float32)
    R_kal = 10.0*np.eye(2).astype(np.float32),
    
    x_est_post_vec = np.zeros((t.size, n_x)).astype(np.float32)
    x_est_pri_vec  = np.zeros((t.size, n_x)).astype(np.float32)

    x_est_pri = x[0, :] # x[0|-1]
    P_pri = np.diag([0.01, 100, 0.01, 100]).astype(np.float32) # P[0|-1]
    I_nx = np.eye(n_x, n_x).astype(np.float32)

    for time_idx in range(len(t)):
        ui = u[time_idx,:]
        yi = y_noise[time_idx,:]

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
    ax[0].plot(t[:n_plot], y_noise[:n_plot, 0], 'k',  label='Measured')
    ax[0].plot(t[:n_plot], x_est_post_vec[:n_plot, 0], label='Predicted')
    ax[0].plot(t[:n_plot], x[:n_plot, 0], label='True')
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

    ax[2].plot(t[:n_plot], y_noise[:n_plot, 1], 'k', label='Measured')
    ax[2].plot(t[:n_plot], x_est_post_vec[:n_plot, 2], label='Predicted')
    ax[2].plot(t[:n_plot], x[:n_plot, 2], label='True')
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
