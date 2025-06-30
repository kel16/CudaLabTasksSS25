import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SimCLR(nn.Module):
    """
    SimCLR model with ResNet-18 Backbone
    """
    def __init__(self, hidden_dim=512, output_dim=128):
        super().__init__()

        # Use a pre-trained ResNet18
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()

        # projection head
        self.proj_head = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        return
    
    def forward(self, x):
        h = self.backbone(x)
        z = self.proj_head(h)
        out = F.normalize(z, dim=1)  # L2 normalize
        return out
    