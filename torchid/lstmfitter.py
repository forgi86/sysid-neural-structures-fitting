from __future__ import print_function
import torch
import torch.nn as nn


class LSTMSimulator(nn.Module):
    def __init__(self, n_input = 1, n_hidden_1 = 64, n_hidden_2 = 32, n_output = 1):

        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_output = n_output

        super(LSTMSimulator, self).__init__()
        self.lstm1 = nn.LSTMCell(self.n_input, self.n_hidden_1) # input size, hidden size
        self.lstm2 = nn.LSTMCell(self.n_hidden_1, self.n_hidden_2)
        self.linear = nn.Linear(self.n_hidden_2, self.n_output)

    def forward(self, input, future = 0):
        batch_size = input.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.n_hidden_1)#, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.n_hidden_1)#, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.n_hidden_2)#, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.n_hidden_2)#, dtype=torch.double)

        seq_len = input.size(1)
        for t in range(seq_len): #, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input[:, t, :]
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # input_t, (hidden_state_t, cell_state_t) -> hidden_state_{t+1}, cell_state_{t+1}
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1)#.squeeze(2)
        return outputs
