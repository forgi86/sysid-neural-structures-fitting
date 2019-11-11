from scipy.integrate import ode
import time
import control
import control.matlab
import pandas as pd
import os
import sys
sys.path.append(os.path.join("..", ".."))
from examples.cartpole_example.cartpole_dynamics import *

Ts_faster_loop = 10e-3
Ts_slower_loop_def = 10e-3

def to_mpipi(x):
    """ Convert an angle to the range (-pi, pi)"""
    x_range = x - 2 * np.pi * ((x + np.pi) // (2 * np.pi))
    return x_range

def to_02pi(x):
    """ Convert an angle to the range (0, 2pi)"""
    x_range = x - 2 * np.pi * ((x) // (2 * np.pi))
    return x_range

DEFAULTS_PENDULUM_MPC = {
    'uref':  np.array([0.0]), # N
    'std_npos': 0*0.001,  # m
    'std_nphi': 0*0.00005,  # rad
    'std_F': 1.5,  # N
    'w_F': 5.0,  # rad
    'len_sim': 80, #s
    'Ts_slower_loop': Ts_slower_loop_def,
    'Q_kal':  np.diag([0.1, 10, 0.1, 10]),
    'R_kal': 1*np.eye(2),
    'seed_val': 42

}


def get_parameter(sim_options, par_name):
    return sim_options.get(par_name, DEFAULTS_PENDULUM_MPC[par_name])


def get_default_parameters(sim_options):
    """ Which parameters are left to default ??"""
    default_keys = [key for key in DEFAULTS_PENDULUM_MPC if key not in sim_options]
    return default_keys


def simulate_pendulum_oloop(sim_options):
    seed_val = get_parameter(sim_options,'seed_val')
    if seed_val is not None:
        np.random.seed(seed_val)

    nx = 4
    ny = 2
    nu = 1

    Ts_slower_loop = get_parameter(sim_options, 'Ts_slower_loop')
    ratio_Ts = int(Ts_slower_loop // Ts_faster_loop)


    # Standard deviation of the measurement noise on position and angle
    std_npos = get_parameter(sim_options, 'std_npos')
    std_nphi = get_parameter(sim_options, 'std_nphi')

    # Force input
    std_F = get_parameter(sim_options, 'std_F')

    # Input force power spectrum
    w_F = get_parameter(sim_options, 'w_F') # bandwidth of the force input
    tau_F = 1 / w_F
    Hu = control.TransferFunction([1], [1 / w_F, 1])
    Hu = Hu * Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts_faster_loop)
    N_sim_imp = tau_F / Ts_faster_loop * 20
    t_imp = np.arange(N_sim_imp) * Ts_faster_loop
    t, y = control.impulse_response(Hud, t_imp)
    y = y[0]
    std_tmp = np.sqrt(np.sum(y ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / (std_tmp) * std_F

    # Input force signal
    N_skip = int(20 * tau_F // Ts_faster_loop) # skip initial samples to get a regime sample of d
    t_sim_d = get_parameter(sim_options, 'len_sim')  # simulation length (s)
    N_sim_d = int(t_sim_d // Ts_faster_loop)
    N_sim_d = N_sim_d + N_skip + 1
    e = np.random.randn(N_sim_d)
    te = np.arange(N_sim_d) * Ts_faster_loop
    _, F_in, _ = control.forced_response(Hu, te, e)
    F_in = F_in.ravel()
    F_in = F_in[N_skip:]

    # Initialize simulation system
    t0 = 0
    phi0 = 180*2*np.pi/360
    x0 = np.array([0, 0, phi0, 0]) # initial state
    system_dyn = ode(f_ODE_wrapped).set_integrator('dopri5')
    system_dyn.set_initial_value(x0, t0)
    system_dyn.set_f_params(0.0)

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
        idx_inner_controller = idx_fast // ratio_Ts
        run_inner_controller = (idx_fast % ratio_Ts) == 0

        y_step = x_step[[0, 2]]
        ymeas_step = np.copy(y_step)
        ymeas_step[0] += std_npos * np.random.randn()
        ymeas_step[1] += std_nphi * np.random.randn()
        y_meas_vec_fast[idx_fast,:] = ymeas_step
        # Output for step i
        # Ts_slower_loop outputs

        if run_inner_controller: # it is also a step of the simulation at rate Ts_slower_loop
            if idx_inner_controller < nsim:
                t_vec[idx_inner_controller, :] = t_step
                y_vec[idx_inner_controller,:] = y_step
                y_meas_vec[idx_inner_controller,:] = ymeas_step
                u_vec[idx_inner_controller, :] = u_PID

        # PID angle CONTROLLER
        ref_angle =  0.0 #angle_ref[idx_fast]
        error_angle = ref_angle - ymeas_step[1]
        u_PID = np.zeros(nu)
        u_PID[u_PID > 10.0] = 10.0
        u_PID[u_PID < -10.0] = -10.0
        u_TOT = 0*u_PID + F_in[idx_fast]

        # Ts_fast outputs
        t_vec_fast[idx_fast,:] = t_step
        x_vec_fast[idx_fast, :] = x_step #system_dyn.y
        u_vec_fast[idx_fast,:] = u_TOT
        Fd_vec_fast[idx_fast,:] = F_in[idx_fast]
        ref_angle_vec_fast[idx_fast,:] = ref_angle

        # Controller simulation step at rate Ts_slower_loop
        if run_inner_controller:
            pass

        # System simulation step at rate Ts_fast
        time_integrate_start = time.perf_counter()
        system_dyn.set_f_params(u_TOT)
        system_dyn.integrate(t_step + Ts_faster_loop)
        x_step = system_dyn.y
        t_int_vec_fast[idx_fast,:] = time.perf_counter() - time_integrate_start

        # Time update
        t_step += Ts_faster_loop

    simout = {'t': t_vec, 'x': x_vec, 'u': u_vec, 'y': y_vec, 'y_meas': y_meas_vec, 'x_ref': x_ref_vec,  'status': status_vec, 'Fd_fast': Fd_vec_fast,
              't_fast': t_vec_fast, 'x_fast': x_vec_fast, 'x_ref_fast': x_ref_vec_fast, 'u_fast': u_vec_fast, 'y_meas_fast': y_meas_vec_fast, 'emergency_fast': emergency_vec_fast,
              'nsim': nsim, 'Ts_slower_loop': Ts_slower_loop, 't_calc': t_calc_vec, 'ref_angle_fast': ref_angle_vec_fast,
              't_int_fast': t_int_vec_fast
              }

    return simout


if __name__ == '__main__':

    # Set seed for reproducibility
    np.random.seed(42)

    import matplotlib.pyplot as plt
    plt.close('all')
    
    simopt = DEFAULTS_PENDULUM_MPC

    time_sim_start = time.perf_counter()
    simout = simulate_pendulum_oloop(simopt)
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

    fig,axes = plt.subplots(3,1, figsize=(10,10), sharex=True)
    #axes[0].plot(t, y_meas[:, 0], "b", label='p_meas')
    axes[0].plot(t_fast, x_fast[:, 0], "k", label='p')
    idx_pred = 0
    #axes[0].set_ylim(-20,20.0)
    axes[0].set_title("Position (m)")


    #axes[1].plot(t, to_02pi(y_meas[:, 1])*RAD_TO_DEG, "b", label='phi_meas')
    axes[1].plot(t_fast, (x_fast[:, 2])*RAD_TO_DEG, 'k', label="phi")
    axes[1].plot(t_fast, ref_phi_fast[:,0]*RAD_TO_DEG, "k--", label="phi_ref")

    idx_pred = 0
    #axes[1].set_ylim(-700, 700)
    axes[1].set_title("Angle (deg)")

    axes[2].plot(t, u[:,0], label="u")
    axes[2].plot(t_fast, F_input, "k", label="Fd")
    axes[2].plot(t, uref*np.ones(np.shape(t)), "r--", label="u_ref")

    axes[2].set_ylim(-20, 20)
    axes[2].set_title("Force (N)")

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


    if not os.path.exists("data"):
        os.makedirs("data")
    df_X.to_csv(os.path.join("data", "pendulum_data_oloop_id.csv"), index=False)
