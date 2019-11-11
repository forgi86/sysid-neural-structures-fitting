import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_response(fs, w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-60, 40)
    ax.set_xlim(0.1, 0.5*fs)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)

"""
fs = 100.0       # Sample rate, Hz
cutoff = 10.0    # Desired cutoff frequency, Hz
trans_width = 20.0  # Width of transition from pass band to stop band, Hz
numtaps = 50      # Size of the FIR filter.
taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
w, h = signal.freqz(taps, [1], worN=2000)
plot_response(fs, w, h, "Low-pass Filter")
"""

add_noise = True


COL_T = ['time']
COL_Y = ['p_meas', 'theta_meas']
COL_X = ['p', 'v', 'theta', 'omega']
COL_U = ['u']
df_X = pd.read_csv(os.path.join("data", "pendulum_data_MPC_ref_id.csv"))

std_noise_p = add_noise * 0.02
std_noise_theta = add_noise * 0.004
std_noise = np.array([std_noise_p, std_noise_theta])

t = np.array(df_X[COL_T], dtype=np.float32)
y = np.array(df_X[COL_Y], dtype=np.float32)
x = np.array(df_X[COL_X], dtype=np.float32)
u = np.array(df_X[COL_U], dtype=np.float32)
y_meas = np.copy(y) + np.random.randn(*y.shape)*std_noise



Ts = np.float(t[1] - t[0])
fs = 1/Ts       # Sample rate, Hz
cutoff = 1.0    # Desired cutoff frequency, Hz
trans_width = 5  # Width of transition from pass band to stop band, Hz
numtaps = 128      # Size of the FIR filter.
taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [2*np.pi*2*np.pi*10*1.5, 0], Hz=fs, type='differentiator')
w, h = signal.freqz(taps, [1], worN=2000)
plot_response(fs, w[1:], h[1:], "Derivative Filter")


plt.figure()
plt.plot(taps)

x_est = np.zeros((y_meas.shape[0], 4), dtype=np.float32)
x_est[:, 0] = y_meas[:, 0]
x_est[:, 2] = y_meas[:, 1]
x_est[:,1] = np.convolve(x_est[:,0],taps, 'same')#signal.lfilter(taps, 1, y_meas[:,0])*2*np.pi
x_est[:,3] = np.convolve(x_est[:,2],taps, 'same')#signal.lfilter(taps, 1, y_meas[:,1])*2*np.pi


fig,ax = plt.subplots(4,1,sharex=True)

ax[0].plot(x[:,0], 'k')
ax[0].plot(x_est[:,0], 'r')

ax[1].plot(x[:,1], 'k')
ax[1].plot(x_est[:,1], 'r')

ax[2].plot(x[:,2], 'k')
ax[2].plot(x_est[:,2], 'r')

ax[3].plot(x[:,3], 'k')
ax[3].plot(x_est[:,3], 'r')
