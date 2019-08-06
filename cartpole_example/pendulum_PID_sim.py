import numpy as np
import scipy.sparse as sparse
from ltisim import LinearStateSpaceSystem
from kalman import kalman_design_simple, LinearStateEstimator
from pendulum_model import *
from scipy.integrate import ode
from scipy.interpolate import interp1d
import time
import control
import control.matlab
import numpy.random
import pandas as pd

Ts_fast = 5e-3

Ac_def = np.array([[0, 1, 0, 0],
               [0, -b / M, -(g * m) / M, (ftheta * m) / M],
               [0, 0, 0, 1],
               [0, b / (M * l), (M * g + g * m) / (M * l), -(M * ftheta + ftheta * m) / (M * l)]])

Bc_def = np.array([
    [0.0],
    [1.0 / M],
    [0.0],
    [-1 / (M * l)]
])

# Reference input and states
#t_ref_vec = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
#p_ref_vec = np.array([0.0, 0.8, 0.8, 0.0, 0.0])
#rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='zero')
t_ref_vec = np.array([0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 100.0])
p_ref_vec = np.array([0.0, 0.0,  0.8, 0.8,  0.0,  0.0,  0.8, 0.8])
rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='linear')


def xref_fun_def(t):
    return np.array([rp_fun(t), 0.0, 0.0, 0.0])


#Qx_def = 0.9 * sparse.diags([0.1, 0, 0.9, 0])   # Quadratic cost for states x0, x1, ..., x_N-1
#QxN_def = Qx_def
#Qu_def = 0.0 * sparse.eye(1)        # Quadratic cost for u0, u1, ...., u_N-1
#QDu_def = 0.01 * sparse.eye(1)       # Quadratic cost for Du0, Du1, ...., Du_N-1

Ts_PID_def = 5e-3#Ts_fast


DEFAULTS_PENDULUM_MPC = {
    'xref_fun': xref_fun_def,
    'uref':  np.array([0.0]), # N
    'std_npos': 0*0.001,  # m
    'std_nphi': 0*0.00005,  # rad
    'std_dF': 1,  # N
    'w_F':1,  # rad
    'len_sim': 40, #s

    'Ac': Ac_def,
    'Bc': Bc_def,
    'Ts_PID': Ts_PID_def,
    'Q_kal':  np.diag([0.1, 10, 0.1, 10]),
    'R_kal': 1*np.eye(2),
    'seed_val': None

}


def get_parameter(sim_options, par_name):
    return sim_options.get(par_name, DEFAULTS_PENDULUM_MPC[par_name])


def get_default_parameters(sim_options):
    """ Which parameters are left to default ??"""
    default_keys = [key for key in DEFAULTS_PENDULUM_MPC if key not in sim_options]
    return default_keys


def simulate_pendulum_MPC(sim_options):

    seed_val = get_parameter(sim_options,'seed_val')
    if seed_val is not None:
        np.random.seed(seed_val)

    Ac = get_parameter(sim_options, 'Ac')
    Bc = get_parameter(sim_options, 'Bc')

    Cc = np.array([[1., 0., 0., 0.],
                   [0., 0., 1., 0.]])

    Dc = np.zeros((2, 1))

    [nx, nu] = Bc.shape  # number of states and number or inputs
    ny = np.shape(Cc)[0]

    Ts_PID = get_parameter(sim_options, 'Ts_PID')
    ratio_Ts = int(Ts_PID // Ts_fast)

    # Brutal forward euler discretization
    Ad = np.eye(nx) + Ac*Ts_PID
    Bd = Bc*Ts_PID
    Cd = Cc
    Dd = Dc

    # Standard deviation of the measurement noise on position and angle

    std_npos = get_parameter(sim_options, 'std_npos')
    std_nphi = get_parameter(sim_options, 'std_nphi')

    # Force disturbance
    std_dF = get_parameter(sim_options, 'std_dF')

    # disturbance power spectrum
    w_F = get_parameter(sim_options, 'w_F') # bandwidth of the force disturbance
    tau_F = 1 / w_F
    Hu = control.TransferFunction([1], [1 / w_F, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts_fast)
    N_sim_imp = tau_F / Ts_fast * 20
    t_imp = np.arange(N_sim_imp) * Ts_fast
    t, y = control.impulse_response(Hud, t_imp)
    y = y[0]
    std_tmp = np.sqrt(np.sum(y ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / (std_tmp) * std_dF


    N_skip = int(20 * tau_F // Ts_fast) # skip initial samples to get a regime sample of d
    t_sim_d = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    N_sim_d = int(t_sim_d // Ts_fast)
    N_sim_d = N_sim_d + N_skip
    e = np.random.randn(N_sim_d)
    te = np.arange(N_sim_d) * Ts_fast
    _, d, _ = control.forced_response(Hu, te, e)
    d_fast = d[N_skip:]
    #td = np.arange(len(d)) * Ts_fast
    

    # Initialize simulation system
    t0 = 0
    phi0 = 10*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    system_dyn = ode(f_ODE_wrapped).set_integrator('vode', method='bdf') #    dopri5
#    system_dyn = ode(f_ODE_wrapped).set_integrator('dopri5')
    system_dyn.set_initial_value(x0, t0)
    system_dyn.set_f_params(0.0)

    # Basic Kalman filter design
    Q_kal =  get_parameter(sim_options, 'Q_kal')
    R_kal = get_parameter(sim_options, 'R_kal')
    L, P, W = kalman_design_simple(Ad, Bd, Cd, Dd, Q_kal, R_kal, type='predictor')
    x0_est = x0
    KF = LinearStateEstimator(x0_est, Ad, Bd, Cd, Dd,L)


    #K_NUM = [0,   -35000000,  -105000000,   -70000000]
    #K_DEN = [1,        2000,     1000000,           0]

    K_NUM = [-2100, -10001, -100]
    K_DEN = [1,   100,     0]

    #K_NUM = [-300,       -1001,         -10]
    #K_DEN = [1,    10,     0]

    K = control.tf(K_NUM,K_DEN)
    Kd_tf = control.c2d(K, Ts_PID)
    Kd_ss = control.ss(Kd_tf)
    Kd = LinearStateSpaceSystem(A=Kd_ss.A, B=Kd_ss.B, C=Kd_ss.C, D=Kd_ss.D)

    # Simulate in closed loop
    len_sim = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    nsim = int(len_sim // Ts_PID) #int(np.ceil(len_sim / Ts_PID))  # simulation length(timesteps) # watch out! +1 added, is it correct?
    t_vec = np.zeros((nsim, 1))
    t_calc_vec = np.zeros((nsim,1)) # computational time to get MPC solution (+ estimator)
    status_vec = np.zeros((nsim,1))
    x_vec = np.zeros((nsim, nx))
    x_ref_vec = np.zeros((nsim, nx))
    y_vec = np.zeros((nsim, ny))
    y_meas_vec = np.zeros((nsim, ny))
    u_vec = np.zeros((nsim, nu))

    nsim_fast = int(len_sim // Ts_fast)
    t_vec_fast = np.zeros((nsim_fast, 1))
    x_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluation
    x_ref_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluatio
    u_vec_fast = np.zeros((nsim_fast, nu)) # finer integration grid for performance evaluatio
    Fd_vec_fast = np.zeros((nsim_fast, nu))  #
    t_int_vec_fast = np.zeros((nsim_fast, 1))
    emergency_vec_fast = np.zeros((nsim_fast, 1))  #


    t_step = t0
    x_step = x0
    u_MPC = None
    for idx_fast in range(nsim_fast):

        ## Determine step type: fast simulation only or MPC step
        idx_MPC = idx_fast // ratio_Ts
        run_controller = (idx_fast % ratio_Ts) == 0

        # Output for step i
        # Ts_PID outputs
        if run_controller: # it is also a step of the simulation at rate Ts_PID
            y_step = Cd.dot(x_step)  # y[i] from the system
            ymeas_step = np.copy(y_step)
            ymeas_step[0] += std_npos * np.random.randn()
            ymeas_step[1] += std_nphi * np.random.randn()
            error_angle = 0 - ymeas_step[1]
            u_MPC = Kd.output(error_angle)
            u_MPC[u_MPC > 10.0] = 10.0
            u_MPC[u_MPC < -10.0] = -10.0

            if idx_MPC < nsim:
                t_vec[idx_MPC, :] = t_step
                y_vec[idx_MPC,:] = y_step
                y_meas_vec[idx_MPC,:] = ymeas_step
                u_vec[idx_MPC, :] = u_MPC

        # Ts_fast outputs
        t_vec_fast[idx_fast,:] = t_step
        x_vec_fast[idx_fast, :] = x_step #system_dyn.y
        u_fast = u_MPC + d_fast[idx_fast]
        u_vec_fast[idx_fast,:] = u_fast
        Fd_vec_fast[idx_fast,:] = d_fast[idx_fast]

        ## Update to step i+1

        # Controller simulation step at rate Ts_PID
        if run_controller:
            time_calc_start = time.perf_counter()
            # Kalman filter: update and predict
            #KF.update(ymeas_step) # \hat x[i|i]
            #KF.predict(u_MPC)    # \hat x[i+1|i]
            KF.predict_update(u_MPC, ymeas_step)
            Kd.update(error_angle)

        # System simulation step at rate Ts_fast
        time_integrate_start = time.perf_counter()
        system_dyn.set_f_params(u_fast)
        system_dyn.integrate(t_step + Ts_fast)
        x_step = system_dyn.y
        #x_step = x_step + f_ODE_jit(t_step, x_step, u_fast)*Ts_fast
        #x_step = x_step + f_ODE(0.0, x_step, u_fast) * Ts_fast
        t_int_vec_fast[idx_fast,:] = time.perf_counter() - time_integrate_start

        # Time update
        t_step += Ts_fast

    simout = {'t': t_vec, 'x': x_vec, 'u': u_vec, 'y': y_vec, 'y_meas': y_meas_vec, 'x_ref': x_ref_vec,  'status': status_vec, 'Fd_fast': Fd_vec_fast,
              't_fast': t_vec_fast, 'x_fast': x_vec_fast, 'x_ref_fast': x_ref_vec_fast, 'u_fast': u_vec_fast, 'emergency_fast': emergency_vec_fast,
              'KF': KF, 'K': K, 'nsim': nsim, 'Ts_PID': Ts_PID, 't_calc': t_calc_vec,
              't_int_fast': t_int_vec_fast
              }

    return simout


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib


    plt.close('all')
    
    simopt = DEFAULTS_PENDULUM_MPC

    time_sim_start = time.perf_counter()
    simout = simulate_pendulum_MPC(simopt)
    time_sim = time.perf_counter() - time_sim_start

    t = simout['t']
    x = simout['x']
    u = simout['u']
    y = simout['y']
    y_meas = simout['y_meas']
    x_ref = simout['x_ref']
    x_fast = simout['x_fast']
    u_fast = simout['u_fast']

    t_fast = simout['t_fast']
    x_ref_fast = simout['x_ref_fast']
    Fd_fast = simout['Fd_fast']
    KF = simout['KF']
    status = simout['status']

    uref = get_parameter(simopt, 'uref')
    nsim = len(t)
    nx = x.shape[1]
    ny = y.shape[1]

    y_ref = x_ref[:, [0, 2]]

    fig,axes = plt.subplots(3,1, figsize=(10,10))
    axes[0].plot(t, y_meas[:, 0], "b", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='p')
    axes[0].plot(t, y_ref[:,0], "k--", label="p_ref")
    idx_pred = 0
    axes[0].set_ylim(-20,20.0)
    axes[0].set_title("Position (m)")


    axes[1].plot(t, y_meas[:, 1]*RAD_TO_DEG, "b", label='phi_meas')
    axes[1].plot(t_fast, x_fast[:, 2]*RAD_TO_DEG, 'k', label="phi")
    idx_pred = 0
    axes[1].set_ylim(-20,20)
    axes[1].set_title("Angle (deg)")

    axes[2].plot(t, u[:,0], label="u")
    axes[2].plot(t_fast, Fd_fast, "k", label="Fd")
    axes[2].plot(t, uref*np.ones(np.shape(t)), "r--", label="u_ref")

    axes[2].set_ylim(-20,20)
    axes[2].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    X = np.hstack((t, x, u, y_meas, Fd_fast))
    COL_T = ['time']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    COL_D = ['d']
    COL_Y = ['p_meas', 'theta_meas']

    COL = COL_T + COL_X + COL_U + COL_Y + COL_D
    df_X = pd.DataFrame(X, columns=COL)
    df_X.to_csv("pendulum_data_PID.csv", index=False)
