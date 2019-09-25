import os
import pandas as pd
import matplotlib.pyplot as plt
from pendulum_model import RAD_TO_DEG, DEG_TO_RAD

if __name__ == '__main__':

    df_model = pd.read_csv(os.path.join("data", "pendulum_data_PID.csv"))
    df_nn = pd.read_csv(os.path.join("data", "pendulum_data_PID_NN_model.csv"))


    fig,axes = plt.subplots(3,1, figsize=(10,10), sharex=True)
    axes[0].plot(df_model['time'], df_model['p'], "k", label='p system')
    axes[0].plot(df_nn['time'], df_nn['p'], "r", label='p NN')
    axes[0].set_title("Position (m)")
    axes[0].set_ylim(-20, 20.0)


    axes[1].plot(df_model['time'], df_model['theta']*RAD_TO_DEG, "k", label='theta system')
    axes[1].plot(df_nn['time'], df_nn['theta']*RAD_TO_DEG, "r", label='theta NN')


    axes[2].plot(df_model['time'], df_model['u'], label="u")
    axes[2].plot(df_nn['time'], df_nn['u'], label="u")


    for ax in axes:
        ax.grid(True)
        ax.legend()
