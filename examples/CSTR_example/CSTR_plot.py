import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys


if __name__ == '__main__':

    COL_T = ['time']
    COL_Y = ['Ca']
    COL_X = ['Ca', 'T']
    COL_U = ['q']

    df_X = pd.read_csv(os.path.join("data", "cstr.dat"), header=None, sep="\t")
    df_X.columns = ['time', 'q', 'Ca', 'T', 'None']


    df_X['q'] = df_X['q']/100
    df_X['Ca'] = df_X['Ca']*10
    df_X['T'] = df_X['T']/400
    
    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)
    x0_torch = torch.from_numpy(x[0,:])


    x_noise = np.copy(x) #+ np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)


    # In[Plot]
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(np.array(x_noise[:,0]), 'k',  label='True')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(np.array(x_noise[:,1]), 'k', label='True')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(u, 'k', label='True')
    ax[2].legend()
    ax[2].grid(True)
