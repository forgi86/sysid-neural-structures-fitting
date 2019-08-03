import os
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=20000)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 100), # 4 states input
            nn.Tanh(),
            nn.Linear(100, 2), # 2 state equations output (2 are trivial!)
        )

        self.AL = nn.Linear(4,4, bias=False)
        self.AL.weight = torch.nn.Parameter(torch.tensor([[0.,1.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]]), requires_grad=False)
        self.WL = nn.Linear(2,4, bias=False)
        self.WL.weight = torch.nn.Parameter(torch.tensor([[0.,0.],[1.,0.],[0.,0.],[0.,1.]]), requires_grad=False)


        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

    def nn_derivative(self, X, U):
        XU = torch.cat((X,U),1)
        FX_TMP = self.net(XU)
        DX = (self.WL(FX_TMP) + self.AL(X))
        return DX

    def forward(self,X,U):
        Ts = 10e-3
        X_pred = torch.empty(X.shape)
        X_pred[0,:] = X[0,:]
        DX = self.nn_derivative(X[0:-1], U[0:-1])
        X_pred[1:,:] = X[0:-1,:] + Ts*DX
        return X_pred

    def f_ODE(self,t,x,u):
        x = torch.tensor(x.reshape(1,-1).astype(np.float32))
        u = torch.tensor(u.reshape(1,-1).astype(np.float32))
        return np.array(self.nn_derivative(x,u)).ravel().astype(np.float64)

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

    df_X = pd.read_csv("pendulum_data.csv")

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y],dtype=np.float32)

    x_true = np.array(df_X[COL_X])
    x_0 = x_true[0,:]
    x_true_torch = torch.tensor(np.array(df_X[COL_X], dtype=np.float32))
    U = torch.tensor(np.array(df_X[COL_U],dtype=np.float32))

    time_torch = torch.from_numpy(time_data.ravel())

    func = ODEFunc()

    C_matrix = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    y_true_torch = torch.tensordot(x_true_torch, C_matrix, ((-1,), (1,)))

    # In[0]
    optimizer = optim.RMSprop(func.net.parameters(), lr=1e-4)
    end = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ii = 0
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        x_pred_torch = func(x_true_torch, U)
        y_pred_torch = torch.tensordot(x_pred_torch, C_matrix, ((-1,), (1,)))
        loss = torch.mean(torch.abs(x_pred_torch - x_true_torch))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                x_pred_torch = func(x_true_torch, U)
                y_pred_torch = torch.tensordot(x_pred_torch, C_matrix, ((-1,), (1,)))
                loss = torch.mean(torch.abs(x_pred_torch - x_true_torch))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        end = time.time()



    u_func = interp1d(time_data.ravel(), np.array(df_X[COL_U]).ravel(), fill_value="extrapolate")

    def f_ODE_sim(x,t):
        u = u_func(t).reshape(-1,1)
        return func.f_ODE(t,x,u)

    with torch.no_grad():
        x_sim = odeint(f_ODE_sim, x_0, time_data.ravel())



    plt.figure()
    plt.plot(np.array(x_true_torch[:,1]))
    plt.plot(np.array(x_pred_torch[:,1]))
    plt.plot(x_sim[:,1],'r')

    plt.figure()
    plt.plot(np.array(x_true_torch[:,3]))
    plt.plot(x_sim[:,3],'r')
