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

    def forward(self, input):
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


class LSTMAutoRegressive(nn.Module):

    def __init__(self, n_input = 1, n_hidden_1 = 64, n_hidden_2 = 32, n_output = 1):
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_output = n_output

        super(LSTMAutoRegressive, self).__init__()
        self.lstm1 = nn.LSTMCell(self.n_input + self.n_output, self.n_hidden_1) # input size, hidden size
        self.lstm2 = nn.LSTMCell(self.n_hidden_1, self.n_hidden_2)
        self.linear = nn.Linear(self.n_hidden_2, self.n_output)

    def forward(self, input, delayed_output): # future=... to predict in the future!
        batch_size = input.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.n_hidden_1)#, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.n_hidden_1)#, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.n_hidden_2)#, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.n_hidden_2)#, dtype=torch.double)

        seq_len = input.size(1)
        for t in range(seq_len): #, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input[:, t, :]
            delayed_output_t = delayed_output[:,t,:]
            feature_t = torch.stack((input_t, delayed_output_t), 1).squeeze(-1)
            h_t, c_t = self.lstm1(feature_t, (h_t, c_t)) # input_t, (hidden_state_t, cell_state_t) -> hidden_state_{t+1}, cell_state_{t+1}
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1)#.squeeze(2)
        return outputs


    def forward_sim(self, input, delayed_output_t=None):
        batch_size = input.size(0)
        outputs = []
        h_t = torch.zeros(batch_size, self.n_hidden_1)#, dtype=torch.double)
        c_t = torch.zeros(batch_size, self.n_hidden_1)#, dtype=torch.double)
        h_t2 = torch.zeros(batch_size, self.n_hidden_2)#, dtype=torch.double)
        c_t2 = torch.zeros(batch_size, self.n_hidden_2)#, dtype=torch.double)

        if delayed_output_t is None:
            delayed_output_t = torch.zeros(batch_size, self.n_output)

        seq_len = input.size(1)
        for t in range(seq_len): #, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input[:, t, :]
            #delayed_output_t = delayed_output[:,t,:]
            feature_t = torch.stack((input_t, delayed_output_t), 1).squeeze(-1)
            h_t, c_t = self.lstm1(feature_t, (h_t, c_t)) # input_t, (hidden_state_t, cell_state_t) -> hidden_state_{t+1}, cell_state_{t+1}
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            delayed_output_t = output
            outputs += [output]
        outputs = torch.stack(outputs, 1)#.squeeze(2)
        return outputs
