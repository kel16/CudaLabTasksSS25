import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # forget gate
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        # input gate
        self.W_i = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        # candidate
        self.W_c = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

        # output gate
        self.W_o = nn.Parameter(torch.randn(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 0.1
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, state):
        h_prev, c_prev = state

        f = torch.sigmoid(x @ self.W_f + h_prev @ self.U_f + self.b_f)

        i = torch.sigmoid(x @ self.W_i + h_prev @ self.U_i + self.b_i)
        g = torch.tanh(x @ self.W_c + h_prev @ self.U_c + self.b_c)

        c = f * c_prev + i * g

        o = torch.sigmoid(x @ self.W_o + h_prev @ self.U_o + self.b_o)
        h = o * torch.tanh(c)
        
        return h, c
