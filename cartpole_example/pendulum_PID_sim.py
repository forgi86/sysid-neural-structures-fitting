import numpy as np
from ltisim import LinearStateSpaceSystem
from pendulum_model import *
from scipy.integrate import ode
from scipy.interpolate import interp1d
import time
import control
import control.matlab
import pandas as pd
import os

k_def = 9
tau_def = 20e-3
Acl_c_def = np.array([[0,1,0], [0, 0, k_def], [0, 0, -1/tau_def]])
Bcl_c_def = np.array([[0],
                      [k_def],
                      [1/tau_def]
                      ])

Ts_faster_loop = 1e-3


t_ref_vec = np.array([0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 100.0])
p_ref_vec = np.array([0.0, 0.0,  0.8, 0.8,  0.0,  0.0,  0.8, 0.8])
rp_fun = interp1d(t_ref_vec, p_ref_vec, kind='linear')


def xref_fun_def(t):
    return np.array([rp_fun(t), 0.0, 0.0, 0.0])

Ts_slower_loop_def = 5e-3 #Ts_fast


DEFAULTS_PENDULUM_MPC = {
    'xref_fun': xref_fun_def,
    'uref':  np.array([0.0]), # N
    'std_npos': 0*0.001,  # m
    'std_nphi': 0*0.00005,  # rad
    'std_dF': 0.05,  # N
    'w_F':20,  # rad
    'len_sim': 40, #s

    'Acl_c': Acl_c_def,
    'Bcl_c': Bcl_c_def,
    'Ts_slower_loop': Ts_slower_loop_def,
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

    Acl_c = get_parameter(sim_options, 'Acl_c')
    Bcl_c = get_parameter(sim_options, 'Bcl_c')

    Ccl_c = np.array([[1., 0., 0],
                      [0., 0., 1]])

    Cc = np.array([[1., 0., 0., 0.],
                   [0., 0., 1., 0.]])

    Dcl_c = np.zeros((2, 1))


    ss_cl_c = control.ss(Acl_c, Bcl_c, Ccl_c, Dcl_c)
    ncl_x, ncl_u = Bcl_c.shape  # number of states and number or inputs
    ncl_y = np.shape(Ccl_c)[0]

    nx, nu = 4,1
    ny = 2

    Ts_slower_loop = get_parameter(sim_options, 'Ts_slower_loop')
    ratio_Ts = int(Ts_slower_loop // Ts_faster_loop)

    # Brutal forward euler discretization

    Acl_d = np.eye(ncl_x) + Acl_c*Ts_slower_loop
    Bcl_d = Bcl_c*Ts_slower_loop
    Ccl_d = Ccl_c
    Dcl_d = Dcl_c

    Cd = Cc

    #ss_cl_d = control.matlab.c2d(Acl_d, Bcl_d, Cd, Dd, Ts_faster_loop)

    # Standard deviation of the measurement noise on position and angle
    std_npos = get_parameter(sim_options, 'std_npos')
    std_nphi = get_parameter(sim_options, 'std_nphi')

    # Force disturbance
    std_dF = get_parameter(sim_options, 'std_dF')

    # Disturbance power spectrum
    w_F = get_parameter(sim_options, 'w_F') # bandwidth of the force disturbance
    tau_F = 1 / w_F
    Hu = control.TransferFunction([1], [1 / w_F, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts_faster_loop)
    N_sim_imp = tau_F / Ts_faster_loop * 20
    t_imp = np.arange(N_sim_imp) * Ts_faster_loop
    t, y = control.impulse_response(Hud, t_imp)
    y = y[0]
    std_tmp = np.sqrt(np.sum(y ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / (std_tmp) * std_dF


    N_skip = int(20 * tau_F // Ts_faster_loop) # skip initial samples to get a regime sample of d
    t_sim_d = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    N_sim_d = int(t_sim_d // Ts_faster_loop)
    N_sim_d = N_sim_d + N_skip + 1
    e = np.random.randn(N_sim_d)
    te = np.arange(N_sim_d) * Ts_faster_loop
    _, d, _ = control.forced_response(Hu, te, e)
    d = d.ravel()
    angle_ref = d[N_skip:]
    

    # Initialize simulation system
    t0 = 0
    phi0 = 10*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    system_dyn = ode(f_ODE_wrapped).set_integrator('vode', method='bdf') #    dopri5
#    system_dyn = ode(f_ODE_wrapped).set_integrator('dopri5')
    system_dyn.set_initial_value(x0, t0)
    system_dyn.set_f_params(0.0)



    #K_NUM = [0,   -35000000,  -105000000,   -70000000]
    #K_DEN = [1,        2000,     1000000,           0]

    K_NUM = [-2100, -10001, -100]
    K_DEN = [1,   100,     0]

    #K_NUM = [-300,       -1001,         -10]
    #K_DEN = [1,    10,     0]

    K = control.tf(K_NUM,K_DEN)
    Kd_tf = control.c2d(K, Ts_slower_loop)
    Kd_ss = control.ss(Kd_tf)
    Kd = LinearStateSpaceSystem(A=Kd_ss.A, B=Kd_ss.B, C=Kd_ss.C, D=Kd_ss.D)

    M_ss = LinearStateSpaceSystem(A=Acl_d, B=Bcl_d, C=Ccl_d, D=Dcl_d)

    # Simulate in closed loop
    len_sim = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    nsim = int(len_sim // Ts_slower_loop) #int(np.ceil(len_sim / Ts_slower_loop))  # simulation length(timesteps) # watch out! +1 added, is it correct?
    t_vec = np.zeros((nsim, 1))
    t_calc_vec = np.zeros((nsim,1)) # computational time to get MPC solution (+ estimator)
    status_vec = np.zeros((nsim,1))
    x_vec = np.zeros((nsim, nx))
    x_ref_vec = np.zeros((nsim, nx))
    y_vec = np.zeros((nsim, ny))
    y_meas_vec = np.zeros((nsim, ny))
    u_vec = np.zeros((nsim, nu))
    x_model_vec = np.zeros((nsim,3))

    nsim_fast = int(len_sim // Ts_faster_loop)
    t_vec_fast = np.zeros((nsim_fast, 1))
    x_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluation
    ref_angle_vec_fast = np.zeros((nsim_fast, 1))
    y_meas_vec_fast = np.zeros((nsim_fast, ny))
    x_ref_vec_fast = np.zeros((nsim_fast, nx)) # finer integration grid for performance evaluatio
    u_vec_fast = np.zeros((nsim_fast, nu)) # finer integration grid for performance evaluatio
    Fd_vec_fast = np.zeros((nsim_fast, nu))  #
    t_int_vec_fast = np.zeros((nsim_fast, 1))
    emergency_vec_fast = np.zeros((nsim_fast, 1))  #


    t_step = t0
    x_step = x0
    u_PID = None
    for idx_fast in range(nsim_fast):

        ## Determine step type: fast simulation only or MPC step
        idx_MPC_controller = idx_fast // ratio_Ts
        run_MPC_controller = (idx_fast % ratio_Ts) == 0

        y_step = Cd.dot(x_step)  # y[i] from the system
        ymeas_step = np.copy(y_step)
        ymeas_step[0] += std_npos * np.random.randn()
        ymeas_step[1] += std_nphi * np.random.randn()
        y_meas_vec_fast[idx_fast,:] = ymeas_step
        # Output for step i
        # Ts_slower_loop outputs

        if run_MPC_controller: # it is also a step of the simulation at rate Ts_slower_loop
            if idx_MPC_controller < nsim:
                t_vec[idx_MPC_controller, :] = t_step
                y_vec[idx_MPC_controller,:] = y_step
                y_meas_vec[idx_MPC_controller,:] = ymeas_step
                u_vec[idx_MPC_controller, :] = u_PID

                ref_angle = angle_ref[idx_MPC_controller]
                model_out = M_ss.output(ref_angle)
                x_model_vec[idx_MPC_controller, :] = M_ss.x.ravel()

        # PID angle CONTROLLER
        ref_angle = angle_ref[idx_fast]
        error_angle = ref_angle - ymeas_step[1]
        u_PID = Kd.output(error_angle)
        u_PID[u_PID > 10.0] = 10.0
        u_PID[u_PID < -10.0] = -10.0
        u_TOT = u_PID


        # Ts_fast outputs
        t_vec_fast[idx_fast,:] = t_step
        x_vec_fast[idx_fast, :] = x_step #system_dyn.y
        u_vec_fast[idx_fast,:] = u_TOT
        Fd_vec_fast[idx_fast,:] = 0.0
        ref_angle_vec_fast[idx_fast,:] = ref_angle

        ## Update to step i+1
        Kd.update(error_angle)

        # Controller simulation step at rate Ts_slower_loop
        if run_MPC_controller:
            M_ss.update(ref_angle)
            pass

        # System simulation step at rate Ts_fast
        time_integrate_start = time.perf_counter()
        system_dyn.set_f_params(u_TOT)
        system_dyn.integrate(t_step + Ts_faster_loop)
        x_step = system_dyn.y
        #x_step = x_step + f_ODE_jit(t_step, x_step, u_TOT)*Ts_fast
        #x_step = x_step + f_ODE(0.0, x_step, u_TOT) * Ts_fast
        t_int_vec_fast[idx_fast,:] = time.perf_counter() - time_integrate_start

        # Time update
        t_step += Ts_faster_loop

    simout = {'t': t_vec, 'x': x_vec, 'u': u_vec, 'y': y_vec, 'y_meas': y_meas_vec, 'x_ref': x_ref_vec,  'status': status_vec, 'Fd_fast': Fd_vec_fast,
              't_fast': t_vec_fast, 'x_fast': x_vec_fast, 'x_ref_fast': x_ref_vec_fast, 'u_fast': u_vec_fast, 'y_meas_fast': y_meas_vec_fast, 'emergency_fast': emergency_vec_fast,
              'K': K, 'nsim': nsim, 'Ts_slower_loop': Ts_slower_loop, 't_calc': t_calc_vec, 'ref_angle_fast': ref_angle_vec_fast, 'x_model': x_model_vec,
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
    y_meas_fast = simout['y_meas_fast']
    u_fast = simout['u_fast']
    x_model = simout['x_model']

    t_fast = simout['t_fast']
    x_ref_fast = simout['x_ref_fast']
    F_input = simout['Fd_fast']
    status = simout['status']
    ref_phi_fast = simout['ref_angle_fast']

    uref = get_parameter(simopt, 'uref')
    nsim = len(t)
    nx = x.shape[1]
    ny = y.shape[1]

    y_ref = x_ref[:, [0, 2]]

    fig,axes = plt.subplots(4,1, figsize=(10,10), sharex=True)
    axes[0].plot(t, y_meas[:, 0], "b", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='p')
    axes[0].plot(t, x_model[:, 0], "r", label='p model')
    axes[0].set_ylim(-30,30.0)
    axes[0].set_title("Position (m)")

    axes[1].plot(t_fast, x_fast[:, 1], "k", label='v')
    axes[1].plot(t, x_model[:, 1], "r", label='v model')
    axes[1].set_ylim(-20,20.0)
    axes[1].set_title("Speed (m/s)")

    axes[2].plot(t, y_meas[:, 1]*RAD_TO_DEG, "b", label='phi_meas')
    axes[2].plot(t_fast, x_fast[:, 2]*RAD_TO_DEG, 'k', label="phi")
    axes[2].plot(t, x_model[:, 2]*RAD_TO_DEG, "r", label='phi model')
    axes[2].plot(t_fast, ref_phi_fast[:,0]*RAD_TO_DEG, "k--", label="phi_ref")
    axes[2].set_ylim(-20,20)
    axes[2].set_title("Angle (deg)")

    axes[3].plot(t, u[:,0], label="u")
    axes[3].plot(t_fast, F_input, "k", label="Fd")
    axes[3].plot(t, uref*np.ones(np.shape(t)), "r--", label="u_ref")
    axes[3].set_ylim(-20,20)
    axes[3].set_title("Force (N)")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    X = np.hstack((t_fast, x_fast, u_fast, y_meas_fast, F_input))
    COL_T = ['time']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    COL_D = ['d']
    COL_Y = ['p_meas', 'theta_meas']

    COL = COL_T + COL_X + COL_U + COL_Y + COL_D
    df_X = pd.DataFrame(X, columns=COL)
    df_X.to_csv(os.path.join("data", "pendulum_data_PID.csv"), index=False)
