import torch
import torch.nn as nn

class NormLayer(nn.Module):
    """ Layer that computer embedding normalization """
    def __init__(self, l=2):
        """ Layer initializer """
        assert l in [1, 2]
        super().__init__()
        self.l = l
        return
    
    def forward(self, x):
        """ Normalizing embeddings x. The shape of x is (B,D) """
        x_normalized = x / torch.norm(x, p=self.l, dim=-1, keepdim=True)
        return x_normalized
    
class ConvBlock(nn.Module):
    """ Building block with 2 convolutions """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        """ """
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    
    def forward(self, x):
        """ """
        y = self.block(x)
        return y

class SiameseModel(nn.Module):
    """ 
    Implementation of a simple siamese model 
    """
    def __init__(self, emb_dim=32, channels=[1, 32, 64, 128], in_spatial=(32, 32)):
        """ Module initializer """
        super().__init__()
        n_layers = len(channels) - 1
        
        # convolutional feature extractor
        cnn = []
        for i in range(n_layers):
            cnn.append( ConvBlock(in_channels=channels[i], out_channels=channels[i+1], kernel_size=3) )
        self.cnn = nn.Sequential(*cnn)
        
        # fully connected embedder
        flat_dim = int(channels[-1] * (in_spatial[0] / (2**n_layers)) * (in_spatial[0] / (2**n_layers)))
        self.fc = nn.Linear(flat_dim, emb_dim)
        
        # auxiliar layers
        self.flatten = nn.Flatten()
        self.norm = NormLayer()
    
        return
    
    def forward_one(self, x):
        """ Forwarding just one sample through the model """
        x = self.cnn(x)
        x_flat = self.flatten(x)
        x_emb = self.fc(x_flat)
        x_emb_norm = self.norm(x_emb)
        return x_emb_norm
    
    def forward(self, anchor, positive, negative):
        """ Forwarding a triplet """
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        
        # is there a more efficient way? anchor (B, C, H, W)
        # all_inputs = torch.cat([anchor, positive, negative], dim=0)  # (3 * B, C, H, W)
        # all_embs = self.forward_one(all_inputs)
        # anchor_emb, posive_emb, negative_emb = all_embs[:B], all_embs[B:2*B], all_embs[2*B:]  # all_embs.chunk(3, dim=0)
        
        return anchor_emb, positive_emb, negative_emb
    