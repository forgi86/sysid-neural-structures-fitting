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

    def f_simerr_minibatch(self, batch_u, batch_y_seq, batch_u_seq):

        batch_size = batch_u.shape[0] # number of training samples in the batch
        seq_len = batch_u.shape[1] # length of the training sequences
        n_a = batch_y_seq.shape[1] # number of autoregressive terms on y
        n_b = batch_u_seq.shape[1] # number of autoregressive terms on u


        Y_pred = torch.empty((batch_size, seq_len, 1))
        for i in range(seq_len):
            phi = torch.cat((batch_y_seq, batch_u_seq), -1)
            yi = self.io_model(phi)
            Y_pred[:, i, :] = yi

            # y shift
            batch_y_seq[:, 1:] = batch_y_seq[:, 0:-1]
            batch_y_seq[:, [0]] = yi[:]

            # u shift
            batch_u_seq[:, 1:] = batch_u_seq[:, 0:-1]
            batch_u_seq[:, [0]] = batch_u[:, i]

        return Y_pred
