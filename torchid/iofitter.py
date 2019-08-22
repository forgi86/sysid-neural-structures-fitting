import torch
import torch.nn as nn
import numpy as np


class NeuralIOSimulator():
    def __init__(self, io_model):
        self.io_model = io_model


    def f_onestep(self, PHI):
        Y_pred = self.io_model(PHI)
        return Y_pred
        pass

    def f_simerr(self, y_seq, u_seq, U):
        N = np.shape(U)[0]
        Y = torch.empty((N, 1))

        for i in range(N):
            phi = torch.cat((y_seq, u_seq))
            yi = self.io_model(phi)
            Y[i,:] = yi

            # y shift
            y_seq[1:] = y_seq[0:-1]
            y_seq[0] = yi

            # u shift
            u_seq[1:] = u_seq[0:-1]
            u_seq[0] = U[i]

        return Y

    def f_simerr_minibatch(self, y_seq_batch, u_seq_batch, U_batch):

        batch_size = U_batch.shape[0] # number of training samples in the batch
        seq_len = U_batch.shape[1] # length of the training sequences
        n_a = y_seq_batch.shape[1] # number of autoregressive terms on y
        n_b = u_seq_batch.shape[1] # number of autoregressive terms on u


        Y_pred = torch.empty((batch_size, 1))
        for i in range(seq_len):
            phi = torch.cat((y_seq_batch, u_seq_batch), -1)
            yi = self.io_model(phi)
            Y_pred[:, i, :] = yi

            # y shift
            y_seq_batch[:,1:] = y_seq_batch[:,0:-1]
            y_seq_batch[:,0] = yi

            # u shift
            u_seq_batch[:,1:] = u_seq_batch[:,0:-1]
            u_seq_batch[:,0] = U_batch[:,i]

        return Y_pred
