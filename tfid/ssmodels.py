import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras

class NeuralStateSpaceModel(tensorflow.keras.Model):

    def __init__(self, n_x, n_u, n_feat=64, init_small=True):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        input_shape = self.n_x + self.n_u
        self.net = tensorflow.keras.Sequential([
            tf.keras.layers.Dense(n_feat, input_shape=(input_shape, ), activation='relu'),
            tf.keras.layers.Dense(n_x, input_shape=(n_feat,)),
            ]
        )

    def call(self, X, U):
        XU = tensorflow.concat((X, U), -1)
        DX = self.net(XU)
        return DX


if __name__ == '__main__':

    import numpy as np
    model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=32)
    n_x = 2
    n_u = 1
    batch_size = 32

    X = np.random.randn(batch_size, n_x).astype(np.float32)
    U = np.random.randn(batch_size, n_u).astype(np.float32)

    DX = model(X,U)
