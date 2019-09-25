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
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class ODEFunc(nn.Module):

    def __init__(self,u_fun):
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
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.u_fun = u_fun

    def forward(self, t, x):
        Ts = 10e-3
        idx = int(t//Ts)
        #print(idx)
        #if idx >= 4000:
        #    idx = 3999
        #ui = self.u[idx]

        ui = torch.tensor(np.array(self.u_fun(t)).reshape(1))
        xu = torch.cat((x, ui), 0)
        fx_tmp = self.net(xu)
        dx = self.WL(fx_tmp) + self.AL(x)
        return dx


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
    x = np.array(df_X[COL_X],dtype=np.float32)
    u = np.array(df_X[COL_U],dtype=np.float32)

    u_fun = interp1d(time_data.ravel(), u.ravel(), kind='linear')

    time_torch = torch.from_numpy(time_data.ravel())
    func = ODEFunc(u_fun)
    x0_torch = torch.from_numpy(x[0,:])
    y_torch = torch.from_numpy(y)

    C_matrix = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    ii = 0
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        pred_x = odeint(func, x0_torch, time_torch)
        pred_y = torch.tensordot(pred_x, C_matrix, ((-1,), (1,)))
        loss = torch.mean(torch.abs(pred_y - y_torch))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_x = odeint(func, x0_torch, time_torch)
                pred_y = torch.tensordot(pred_x, C_matrix, ((-1,), (1,)))
                loss = torch.mean(torch.abs(pred_y - y_torch))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        end = time.time()

