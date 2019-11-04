import os
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class ODEFunc(nn.Module):

    def __init__(self,u):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 50), # 4 states input
            nn.Tanh(),
            nn.Linear(50, 2), # 2 state equations output (2 are trivial!)
        )

        self.AL = nn.Linear(4,4, bias=False)
        self.AL.weight = torch.nn.Parameter(torch.tensor([[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]]), requires_grad=False)
        self.WL = nn.Linear(2,4, bias=False)
        self.WL.weight = torch.nn.Parameter(torch.tensor([[0.,0.],[1.,0.],[0.,0.],[0.,1.]]), requires_grad=False)


        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

        self.u = torch.Tensor(u)

    def forward(self, x0):
        Ts = 5e-3
        N = np.shape(self.u)[0]
        nx = np.shape(x0)[0]

        X = torch.empty((N,nx))
        xstep = x0
        for i in range(0,N):
            X[i,:] = xstep

            #identity = xold
            ustep = self.u[i]
            #uold = torch.tensor([0.0])
            xu = torch.cat((xstep, ustep), 0)
            fx_tmp = self.net(xu)
            dx = Ts*(self.WL(fx_tmp) + self.AL(xstep))
            xstep = xstep + dx

        return X

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


if __name__ == '__main__':
    COL_T = ['time']
    COL_Y = ['p_meas', 'theta_meas']
    COL_X = ['p', 'v', 'theta', 'omega']
    COL_U = ['u']
    COL_D = ['d']

    df_X = pd.read_csv("pendulum_data_PID.csv")

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)
    x = np.array(df_X[COL_X],dtype=np.float32)

    d = np.array(df_X[COL_D],dtype=np.float32)
    d_fun = interp1d(time_data.ravel(), d.ravel(), kind='linear')

    time_torch = torch.from_numpy(time_data.ravel())
    func = ODEFunc(d)
    x0_torch = torch.from_numpy(x[0,:])
    y_true_torch = torch.from_numpy(y)
    C_matrix = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
 

    # In[0]
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ii = 0
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        x_pred_torch = func(x0_torch)
        y_pred_torch = torch.tensordot(x_pred_torch, C_matrix, ((-1,), (1,)))
        loss = torch.mean(torch.abs(y_pred_torch - y_true_torch))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                x_pred_torch = func(x0_torch)
                y_pred_torch = torch.tensordot(x_pred_torch, C_matrix, ((-1,), (1,)))
                loss = torch.mean(torch.abs(y_pred_torch - y_true_torch))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        end = time.time()

    # In[1]
    plt.figure()
    plt.plot(np.array(y_true_torch[:,0]))
    plt.plot(np.array(y_pred_torch[:,0]))

    plt.figure()
    plt.plot(np.array(y_true_torch[:,1]))
    plt.plot(np.array(y_pred_torch[:,1]))
