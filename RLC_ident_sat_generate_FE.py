from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import control
import control.matlab
import pandas as pd


from symbolic_RLC import fxu_ODE, fxu_ODE_mod

if __name__ == '__main__':

    np.random.seed(42)
    # Input characteristics #
    len_sim = 5e-3
    Ts = 2e-7
    
    omega_input = 150e3
    std_input = 80
    
    tau_input = 1/omega_input
    Hu = control.TransferFunction([1], [1 / omega_input, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts)

    N_sim = int(len_sim//Ts)
    N_skip = int(20 * tau_input // Ts) # skip initial samples to get a regime sample of d
    N_sim_u = N_sim + N_skip
    e = np.random.randn(N_sim_u)
    te = np.arange(N_sim_u) * Ts
    _, u, _ = control.forced_response(Hu, te, e)
    u = u[N_skip:]
    u = u /np.std(u) * std_input
    
    t_sim = np.arange(N_sim) * Ts
    u_func = interp1d(t_sim, u, kind='zero', fill_value="extrapolate")


    def f_ODE(t,x):
        u = u_func(t).ravel()
        return fxu_ODE(t, x, u)

    def f_ODE_mod(t,x):
        u = u_func(t).ravel()
        return fxu_ODE_mod(t, x, u)


    x0 = np.zeros(2)
    t_span = (t_sim[0],t_sim[-1])

    x1 = np.empty((len(t_sim), x0.shape[0]))
    x2 = np.empty((len(t_sim), x0.shape[0]))

    x1step = np.copy(x0)
    x2step = np.copy(x0)
    for idx in range(len(t_sim)):
        time = t_sim[idx]
        x1[idx,:] = x1step
        x2[idx,:] = x2step
        x1step += f_ODE(time, x1step)*Ts
        x2step += f_ODE_mod(time, x2step)*Ts

    y1_RK = solve_ivp(f_ODE, t_span, x0, t_eval = t_sim)
    y2_RK = solve_ivp(f_ODE_mod, t_span, x0, t_eval = t_sim)
    
    x1_RK = y1_RK.y.T
    x2_RK = y2_RK.y.T
    
    # In[plot]
    fig, ax = plt.subplots(3,1, figsize=(10,10), sharex=True)
    ax[0].plot(t_sim, x1[:,0],'b')
    ax[0].plot(t_sim, x1_RK[:,0],'b*')
    ax[0].plot(t_sim, x2[:,0],'r')
    ax[0].plot(t_sim, x2_RK[:,0],'r*')
    ax[0].set_xlabel('time (s')
    ax[0].set_ylabel('Capacitor voltage (V)')
    
    ax[1].plot(t_sim, x1[:,1],'b')
    ax[1].plot(t_sim, x1_RK[:,1],'b*')
    ax[1].plot(t_sim, x2[:,1],'r')
    ax[1].plot(t_sim, x2_RK[:,1],'r*')
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('Inductor current (A)')
    
    ax[2].plot(t_sim, u,'b')
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('Input voltage (V)')

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)

    X = np.hstack((t_sim.reshape(-1, 1), x1, u.reshape(-1, 1), x1[:, 0].reshape(-1, 1)))
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    COL = COL_T + COL_X + COL_U + COL_Y
    df_X = pd.DataFrame(X, columns=COL)
    df_X.to_csv("RLC_data_FE.csv", index=False)

    X = np.hstack((t_sim.reshape(-1, 1), x2, u.reshape(-1, 1), x2[:, 0].reshape(-1, 1)))
    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    COL = COL_T + COL_X + COL_U + COL_Y
    df_X = pd.DataFrame(X, columns=COL)
    df_X.to_csv("RLC_data_sat_FE.csv", index=False)
