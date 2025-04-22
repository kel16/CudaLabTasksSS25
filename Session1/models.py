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
        
        self.flatten = nn.Flatten()
        layers = nn.Sequential()
        layers.add_module("dense1", nn.Linear(in_features=input_dim, out_features=hidden_dim))
        layers.add_module("act1", nn.ReLU())
        if use_dropout:
            layers.add_module("drop1", nn.Dropout(0.2))
        layers.add_module("dense2", nn.Linear(hidden_dim, hidden_dim))
        layers.add_module("act2", nn.ReLU())
        if use_dropout:
            layers.add_module("drop2", nn.Dropout(0.5))
        layers.add_module("dense3", nn.Linear(hidden_dim, hidden_dim))
        layers.add_module("act3", nn.ReLU())
        if use_dropout:
            layers.add_module("drop3", nn.Dropout(0.5))
        layers.add_module("output", nn.Linear(hidden_dim, output_dim))
        # layers.add_module("outact", nn.Sigmoid())

        self.layers = layers
        
    def forward(self, x):
        """ Forward pass through the model"""
        flatten_input = self.flatten(x)
        assert len(flatten_input.shape) == 2, f"ERROR! Shape of input must be 2D (b_size, dim)"
        pred = self.layers(flatten_input)
        
        return pred

class CNN(nn.Module):
    """ 
    Varation of LeNet: a simple CNN model
    for CIFAR-10 dataset
    """
    def __init__(self, input_dim, output_dim, use_dropout=False):
        """ Model initializer """
        super().__init__()
        
        # layer 1
        # (3, 32, 32)
        conv1 = nn.Conv2d(in_channels=input_dim[-1], out_channels=input_dim[-2], kernel_size=5, stride=1, padding=2)
        # (32, 32, 32)
        relu1 = nn.ReLU()
        dropout1 = nn.Dropout2d(p=0.25) if use_dropout else nn.Identity()
        maxpool1 = nn.MaxPool2d(kernel_size=2)
        # (32, 16, 16)
        
        self.layer1 = nn.Sequential(
                conv1, relu1, dropout1, maxpool1
            )
          
        # layer 2
        # (32, 16, 16)
        conv2 = nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=5, stride=1, padding=2)
        # (32, 16, 16)
        relu2 = nn.ReLU()
        dropout2 = nn.Dropout2d(p=0.5) if use_dropout else nn.Identity()
        maxpool2 = nn.MaxPool2d(kernel_size=2)
        # (32, 8, 8)
        self.layer2 = nn.Sequential(
                conv2, relu2, dropout2, maxpool2
            )
        
        # layer 3
        # (32, 8, 8)
        conv3 = nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=5, stride=1, padding=2)
        # (64, 8, 8)
        relu3 = nn.ReLU()
        dropout3 = nn.Dropout2d(p=0.5) if use_dropout else nn.Identity()
        maxpool3 = nn.MaxPool2d(kernel_size=2)
        # (64, 4, 4)
        self.layer3 = nn.Sequential(
                conv3, relu3, dropout3, maxpool3
            )
        
        # fully connected layer 1
        in_dim = 64 * 4 * 4
        hidden_dim = 64
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        relu4 = nn.ReLU()
        dropout4 = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.fc_layer1 = nn.Sequential(
                self.fc1, relu4, dropout4
            )
        # fully connected layer 2
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.fc_layer2 = nn.Sequential(
                self.fc2
            )
        
    def forward(self, x):
        """ Forward pass """
        cur_b_size = x.shape[0]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2) 
        out3_flat = out3.view(cur_b_size, -1)
        
        out4_flat = self.fc_layer1(out3_flat)
        y = self.fc_layer2(out4_flat)
        return y
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(input_dim, depth, kernel_size=9, patch_size=4, n_classes=10):
    """ 
    Modified implementation from https://arxiv.org/pdf/2201.09792
    """
    return nn.Sequential(nn.Conv2d(3, input_dim[-1], kernel_size=patch_size, stride=patch_size),
                         nn.GELU(), nn.BatchNorm2d(input_dim[-1]),
                         *[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(input_dim[-1], input_dim[-1], 
                                                                          kernel_size, groups=input_dim[-1], padding="same"),
                                                                nn.GELU(), nn.BatchNorm2d(input_dim[-1]))),
                                         nn.Conv2d(input_dim[-1], input_dim[-1], kernel_size=1), nn.GELU(), 
                                         nn.BatchNorm2d(input_dim[-1])) 
                           for i in range(depth)],
                         nn.AdaptiveAvgPool2d((1,1)),
                         nn.Flatten(),
                         nn.Linear(input_dim[-1], n_classes))


    
def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params
