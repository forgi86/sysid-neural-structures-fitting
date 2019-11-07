import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == '__main__':

    # Column names
    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    # Load dataset
    df_X = pd.read_csv(os.path.join("data", "cstr.dat"), header=None, sep="\t")
    df_X.columns = ['time', 'q', 'Ca', 'T', 'None']

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)

    # Plot data in original scale
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(time_data, np.array(x[:, 0]), 'k',  label='$C_a$')
    ax[0].grid(True)
    ax[0].set_ylabel("Concentration (mol/l)")
    ax[0].legend(loc='upper right')

    ax[1].plot(time_data, np.array(x[:, 1]), 'k', label='$T$')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_ylabel("Temperature (K)")
    ax[1].legend(loc='upper right')

    ax[2].plot(time_data, u, 'k', label='$q$')
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_xlabel("Time (min)")
    ax[2].set_ylabel("Flow (l/min)")
    ax[2].legend(loc='upper right')


    # Rescale data
    Ts = time_data[1] - time_data[0]
    t_fit = 500  # Fit on first 500 minutes
    n_fit = int(t_fit//Ts)
    df_id = df_X.iloc[0:n_fit].copy()
    df_val = df_X.iloc[n_fit:].copy()

    COLUMNS_SCALE = ['Ca', 'T', 'q'] # columns to be scaled
    scaler = StandardScaler().fit(df_id[COLUMNS_SCALE])

    # scale identification dataset
    df_id[COLUMNS_SCALE] = scaler.transform(df_id[COLUMNS_SCALE])
    df_id[COL_T] = df_id[COL_T] - df_id[COL_T].iloc[0]

    # scale fit dataset
    df_val[COLUMNS_SCALE] = scaler.transform(df_val[COLUMNS_SCALE])
    df_val[COL_T] = df_val[COL_T] - df_val[COL_T].iloc[0]

    # save datasets to csv
    df_id.to_csv(os.path.join("data", "CSTR_data_id.csv"),  sep=",", index=False)
    df_val.to_csv(os.path.join("data", "CSTR_data_val.csv"),  sep=",", index=False)

    # save scaler object
    with open(os.path.join("data", "fit_scaler.pkl"), 'wb') as fp:
        pickle.dump(scaler, fp)
