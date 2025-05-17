import torch
from torch import nn

class SequentialClassifier(nn.Module):
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
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, emb_dim, 3, 1, 1),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        
        # LSTM model
        self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        
        # FC-classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=4)
        
        return
    
    
    def forward(self, x):
        """ Forward pass through model """
        
        b_size, num_frames, n_channels, n_rows, n_cols = x.shape
        h, c = self.init_state(b_size=b_size, device=x.device) 
        
        # encoding all images in parallel rows
        x = x.view(b_size * num_frames, n_channels, n_rows, n_cols)
        embeddings = self.encoder(x)  # (b*T, C, H, W)  --> (B * T, emb_dim)
        embeddings = embeddings.reshape(b_size, num_frames, -1)

        # feeding LSTM. Does everything for you
        lstm_out, (h_out, c_out) = self.lstm(embeddings, (h, c)) 
        
        # classifying
        y = self.classifier(lstm_out[:, -1, :])  # feeding only output at last layer
        
        return y
    
        
    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = torch.zeros(self.num_layers, b_size, self.hidden_dim)
            c = torch.zeros(self.num_layers, b_size, self.hidden_dim)
        elif(self.mode == "random"):
            h = torch.randn(self.num_layers, b_size, self.hidden_dim)
            c = torch.randn(self.num_layers, b_size, self.hidden_dim)
        h = h.to(device)
        c = c.to(device)
        
        return h, c
    