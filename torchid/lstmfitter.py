from __future__ import print_function
import torch
import torch.nn as nn

class LSTMSimulator(nn.Module):
    def __init__(self):
        super(LSTMSimulator, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51) # input size, hidden size
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51)#, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51)#, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51)#, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51)#, dtype=torch.double)

        for i in range(input.size(1)): #, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input[:, i, :]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # input_t, (hidden_state_t, cell_state_t) -> hidden_state_{t+1}, cell_state_{t+1}
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1)#.squeeze(2)
        return outputs
