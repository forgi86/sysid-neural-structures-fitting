import numpy as np


def r_square(y_pred, y_true, w=1, time_axis=0):
    """ Get the r-square fit criterion per time signal """
    SSE = np.sum((y_pred - y_true)**2, axis=time_axis)
    y_mean = np.mean(y_true, axis=time_axis)
    SST = np.sum((y_true - y_mean)**2, axis=time_axis)

    return 1.0 - SSE/SST


def error_rmse(y_pred, y_true, time_axis=0):
    """ Compute the Root Mean Square Error (RMSE) per time signal """

    SSE = np.mean((y_pred - y_true)**2, axis=time_axis)
    RMS = np.sqrt(SSE)
    return RMS


if __name__ == '__main__':
    N = 20
    ny = 2
    SNR = 10
    y_true = SNR*np.random.randn(N,2)
    y_pred = np.copy(y_true) + np.random.randn(N,2)
    r_square(y_true, y_pred)
