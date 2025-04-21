import torch.nn as nn

class MLP(nn.Module):
    """
    MLP composed of two fully connected layers.
     - First layer takes pixel values and maps them to a hidden dimension
     - Nonlinear activation
     - Third layer maps from hidden dimension to number of classes, predicting a score for each of the classes
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_dropout=False):
        """ Model initalizer """
        super().__init__()

        layers = nn.Sequential()
        layers.add_module("dense1", nn.Linear(in_features=input_dim, out_features=hidden_dim))
        layers.add_module("act1", nn.ReLU())
        if use_dropout:
            layers.add_module("drop1", nn.Dropout(0.2))
        layers.add_module("dense2", nn.Linear(hidden_dim, hidden_dim))
        layers.add_module("act2", nn.ReLU())
        if use_dropout:
            layers.add_module("drop2", nn.Dropout(0.5))
        # An additional layer gave no improvement in accuracy
        # layers.add_module("dense3", nn.Linear(hidden_dim, hidden_dim))
        # layers.add_module("act3", nn.ReLU())
        # if use_dropout:
        #     layers.add_module("drop3", nn.Dropout(0.5))
        layers.add_module("output", nn.Linear(hidden_dim, output_dim))
        # layers.add_module("outact", nn.Sigmoid())

        self.layers = layers
        
    def forward(self, x):
        """ Forward pass through the model"""
        flattened_x = x.flatten(start_dim=1)
        assert len(flattened_x.shape) == 2, f"ERROR! Shape of input must be 2D (b_size, dim)"
        pred = self.layers(flattened_x)
        
        return pred

    def count_model_params(model):
        """ Counting the number of learnable parameters in a nn.Module """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return num_params
