import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(".."))
from tfid.ssfitter import NeuralStateSpaceSimulator
from tfid.ssmodels import NeuralStateSpaceModel

if __name__ == '__main__':

    num_iter = 40000
    test_freq = 10

    add_noise = False

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    df_X = pd.read_csv(os.path.join("data", "RLC_data_id.csv"))

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    std_noise_V = add_noise * 10.0
    std_noise_I = add_noise * 1.0
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)

    Ts = time_data[1] - time_data[0]
    t_fit = 2e-3
    n_fit = int(t_fit // Ts)  # x.shape[0]

    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    u_fit = u[0:n_fit]
    x_targ_fit = x_noise[0:n_fit]
    x_est_init = nn_solution(x_targ_fit, u_fit)


    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    compile_start = time.time()
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            x_est = nn_solution.f_sim(x_noise[0,:], u_fit)
            loss = loss_object(x_est, x_targ_fit)
        gradients = tape.gradient(loss, nn_solution.trainable_variables)
        optimizer.apply_gradients(zip(gradients, nn_solution.trainable_variables))
        return loss
    loss = train_step()
    compile_time = time.time() - compile_start
    print(f"\nCompile time: {compile_time:.2f}")

    train_start = time.time()
    LOSS = []
    for itr in range(num_iter):
        loss=train_step()
        LOSS.append(np.float32(loss))
        if itr % test_freq == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss))

    train_time = time.time() - train_start
    print(f"\nTrain time: {train_time:.2f}")

    x_sim = nn_solution.f_sim(x[0, :], u_fit)

    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(x_targ_fit[:,0], 'k',  label='True')
    ax[0].plot(x_sim[:,0],'r', label='Sim')
    ax[0].legend()
    ax[1].plot(x_targ_fit[:,1], 'k',  label='True')
    ax[1].plot(x_sim[:,1],'r', label='Sim')
    ax[0].grid(True)
    ax[1].grid(True)

    fig, ax = plt.subplots(1,1,sharex=True)
    ax.plot(LOSS)
