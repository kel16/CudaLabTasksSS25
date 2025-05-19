import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_f = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))

        self.W_i = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))

        self.W_C = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_C = nn.Parameter(torch.zeros(hidden_size))

        self.W_o = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, state):
        h_prev, C_prev = state
        combined = torch.cat((h_prev, x), dim=1)  #[batch_size, hidden + input]

        f_t = torch.sigmoid(combined @ self.W_f.T + self.b_f)
        i_t = torch.sigmoid(combined @ self.W_i.T + self.b_i)
        C_tilde = torch.tanh(combined @ self.W_C.T + self.b_C)
        C_t = f_t * C_prev + i_t * C_tilde
        o_t = torch.sigmoid(combined @ self.W_o.T + self.b_o)
        h_t = o_t * torch.tanh(C_t)
        
        return h_t, C_t
    
    
class MySequentialClassifier(nn.Module):
    """ 
    Sequential classifier. Embedded images are fed to a RNN
    
    Args:
    -----
    emb_dim: integer 
        dimensionality of the vectors fed to the LSTM
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        number of stacked LSTMS
    mode: string
        intialization of the states
    """
    
    def __init__(self, emb_dim, hidden_dim, num_layers=1, mode="zeros"):
        """ Module initializer """
        assert mode in ["zeros", "random"]
        super().__init__()
        self.hidden_dim =  hidden_dim
        self.num_layers = num_layers
        self.mode = mode

        # for embedding rows into vector representations
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),       # [B, 3, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # → [B, 64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),     # → [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # → [B, 128, 16, 16]
            nn.Dropout2d(0.2),

            nn.Conv2d(128, emb_dim, kernel_size=3, padding=1),# → [B, emb_dim, 16, 16]
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),                     # → [B, emb_dim, 1, 1]
            nn.Flatten()                                      # → [B, emb_dim]
        )
        
        # LSTM model       
        lstms = []
        for i in range(num_layers):
            in_size = emb_dim if i == 0 else hidden_dim
            lstms.append( MyLSTMCell(input_size=in_size, hidden_size=hidden_dim))
        self.lstm = nn.ModuleList(lstms)
        
        # FC-classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=6)
        
        return
    
    
    def forward(self, x):
        """ Forward pass through model """
        
        b_size, num_frames, n_channels, n_rows, n_cols = x.shape
        h, c = self.init_state(b_size=b_size, device=x.device) 
        
        # embedding rows
        x = x.view(b_size * num_frames, n_channels, n_rows, n_cols)
        embeddings = self.encoder(x)
        embeddings = embeddings.reshape(b_size, num_frames, -1)
        
        # iterating over sequence length
        lstm_out = []
        for i in range(embeddings.shape[1]):  # iterate over time steps
            lstm_input = embeddings[:, i, :]
            # iterating over LSTM Cells
            for j, lstm_cell in enumerate(self.lstm):
                h[j], c[j] = lstm_cell(lstm_input, (h[j], c[j]))
                lstm_input = h[j]
            lstm_out.append(lstm_input)
        lstm_out = torch.stack(lstm_out, dim=1)
            
        # classifying
        y = self.classifier(lstm_out[:, -1, :])  # feeding only output at last layer
        
        return y
    
        
    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        elif(self.mode == "random"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        return h, c
    

class SequentialClassifier(nn.Module):
    """ 
    Sequential classifier. Embedded images are fed to a RNN
    Same as above, but using nn.LSTMCells from PyTorch
    
    Args:
    -----
    emb_dim: integer 
        dimensionality of the vectors fed to the LSTM
    hidden_dim: integer
        dimensionality of the states in the cell
    num_layers: integer
        number of stacked LSTMS
    mode: string
        intialization of the states
    """
    
    def __init__(self, emb_dim, hidden_dim, num_layers=1, mode="zeros"):
        """ Module initializer """
        assert mode in ["zeros", "random"]
        super().__init__()
        self.hidden_dim =  hidden_dim
        self.num_layers = num_layers
        self.mode = mode

        # for embedding rows into vector representations
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),       # [B, 3, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # → [B, 64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),     # → [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # → [B, 128, 16, 16]
            nn.Dropout2d(0.2),

            nn.Conv2d(128, emb_dim, kernel_size=3, padding=1),# → [B, emb_dim, 16, 16]
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),                     # → [B, emb_dim, 1, 1]
            nn.Flatten()                                      # → [B, emb_dim]
        )
        
        # LSTM model       
        lstms = []
        for i in range(num_layers):
            in_size = emb_dim if i == 0 else hidden_dim
            lstms.append( nn.LSTMCell(input_size=in_size, hidden_size=hidden_dim) )
        self.lstm = nn.ModuleList(lstms)
        
        # FC-classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=6)
        
        return
    
    
    def forward(self, x):
        """ Forward pass through model """
        
        b_size, num_frames, n_channels, n_rows, n_cols = x.shape
        h, c = self.init_state(b_size=b_size, device=x.device) 
        
        # embedding rows
        x = x.view(b_size * num_frames, n_channels, n_rows, n_cols)
        embeddings = self.encoder(x)
        embeddings = embeddings.reshape(b_size, num_frames, -1)
        
        # iterating over sequence length
        lstm_out = []
        for i in range(embeddings.shape[1]):  # iterate over time steps
            lstm_input = embeddings[:, i, :]
            # iterating over LSTM Cells
            for j, lstm_cell in enumerate(self.lstm):
                h[j], c[j] = lstm_cell(lstm_input, (h[j], c[j]))
                lstm_input = h[j]
            lstm_out.append(lstm_input)
        lstm_out = torch.stack(lstm_out, dim=1)
            
        # classifying
        y = self.classifier(lstm_out[:, -1, :])  # feeding only output at last layer
        
        return y
    
        
    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        elif(self.mode == "random"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        return h, c
    