import torch
import torch.nn as nn
import torchvision.models as models

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

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        # until AvgPool
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, C, H, W]
        x = self.avgpool(x)            # [B, C, 1, 1]
        x = torch.flatten(x, 1)        # [B, C]
        return x

class SiameseModel(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.backbone = ResNetBackbone()
        in_features = 512
        self.fc = nn.Linear(in_features, emb_dim)
        self.norm = NormLayer()

    def forward_one(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.norm(x)
        return x

    def forward(self, anchor, positive, negative):
        # anchor_emb = self.forward_one(anchor)
        # positive_emb = self.forward_one(positive)
        # negative_emb = self.forward_one(negative)

        all_inputs = torch.cat([anchor, positive, negative], dim=0)  # (3 * B, C, H, W)
        all_embs = self.forward_one(all_inputs)                      # (3 * B, D)
        anchor_emb, positive_emb, negative_emb = torch.chunk(all_embs, 3, dim=0)

        return anchor_emb, positive_emb, negative_emb
