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
