import torch
import numpy as np

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_torch_regressor_mat(x, n_a):
    seq_len = x.shape[0]
    X = torch.empty((seq_len - n_a + 1, n_a))
    for idx in range(seq_len - n_a + 1):
        X[idx] = x[idx:idx + n_a].flip([0])
    return X


if __name__ == '__main__':

    N = 10
    n_a = 3
    x = torch.tensor(np.arange(N))

    X = torch.empty((N - n_a + 1, n_a))
    for idx in range(N - n_a + 1):
        X[idx] = x[idx:idx + n_a].flip([0])
