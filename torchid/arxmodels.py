import torch
import torch.nn as nn
import numpy as np

class NeuralArxModel(nn.Module):
    def __init__(self, n_a, n_b, n_feat=64):
        super(NeuralArxModel, self).__init__()
        self.n_a = n_a
        self.n_b = n_b
        self.n_feat = 64

        self.const = torch.zeros((n_a + n_b,1))
        self.const[0,0] = 1.0

        self.net = nn.Sequential(
            nn.Linear(n_a + n_b, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, 1)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-4)
                nn.init.constant_(m.bias, val=0)

    def forward(self, phi):
        Y = self.net(phi) + torch.matmul(phi, self.const)
        return Y
