import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras
import numpy as np

        
class NeuralStateSpaceSimulator(tensorflow.keras.Model):
    def __init__(self, ss_model):
        super(NeuralStateSpaceSimulator, self).__init__()
        self.ss_model = ss_model

    def call(self, X, U):
        DX = self.ss_model(X[0:-1], U[0:-1])
        X_pred = X[0:-1,:] + DX
        X_pred = tf.concat([tf.reshape(X[0, :], (1, -1)), X_pred], axis=0)
        return X_pred


    def f_sim(self, x0, u):
        N = np.shape(u)[0]

        X_list = []
        xstep = tf.reshape(x0, (1,-1))
        for i in range(N):
            X_list += [xstep]
            # X[i,:] = xstep
            ustep = u[i]
            dx = self.ss_model(tf.reshape(xstep, (1,-1)), tf.reshape(ustep,(1,-1)))
            xstep = xstep + dx

        X = tf.squeeze(tf.stack(X_list, 0))  # .squeeze(2)
        return X

    """
    def f_sim_minibatch(self, x0_batch, U_batch):
        batch_size = x0_batch.shape[0]
        n_x = x0_batch.shape[1]
        seq_len = U_batch.shape[1]

        X_pred_list = []#X_pred = torch.empty((batch_size, seq_len, n_x))
        xstep = x0_batch
        for i in range(seq_len):
            X_pred_list += [xstep] #X_pred[:, i, :] = xstep
            ustep = U_batch[:, i, :]
            dx = self.ss_model(xstep, ustep)
            xstep = xstep + dx

        X_pred = tf.stack(X_pred_list, 1)#.squeeze(2)
        return X_pred
    """
