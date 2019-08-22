import torch
import torch.nn as nn
import numpy as np


class NeuralARXSimulator():
    def __init__(self, arx_model):
        self.arx_model = arx_model


    def f_onestep(self, PHI):
        Y_pred = self.arx_model(PHI)
        return Y_pred
        pass

    def f_simerr(self, U, y_seq, u_seq):
        N = np.shape(U)[0]
        Y = torch.empty((N, 1))

        for i in range(N):
            phi = torch.cat((y_seq, u_seq))
            yi = self.arx_model(phi)
            Y[i,:] = yi

            # y shift
            y_seq[1:] = y_seq[0:-1]
            y_seq[0] = yi

            # u shift
            u_seq[1:] = u_seq[0:-1]
            u_seq[0] = U[i]

        return Y
