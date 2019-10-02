import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.arange(-4*np.pi, 4*np.pi, 0.1)

    x_range = x - 2*np.pi*((x+np.pi)//(2*np.pi))


    plt.plot(x)
    plt.plot(x_range,'r')
