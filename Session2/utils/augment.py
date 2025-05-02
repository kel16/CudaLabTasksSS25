import torch
import numpy as np


class CutMix:
    """
    CutMix data augmentation as described in:
    Yun et al. (2019): "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    https://arxiv.org/pdf/1905.04899
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)
        targets = torch.tensor(targets)
        
        N, _, H, W = inputs.size()

        indices = torch.randperm(N)
        shuffled_inputs = inputs[indices]
        shuffled_targets = targets[indices]
        
        lam = np.random.uniform(0, 1)
        r_x = np.random.uniform(0, W)
        r_y = np.random.uniform(0, H)

        r_w = W * np.sqrt(1 - lam)
        r_h = H * np.sqrt(1 - lam)
        x1 = int(np.round(np.clip(r_x - r_w / 2, 0, None)))   # Clip min=0
        x2 = int(np.round(np.clip(r_x + r_w / 2, None, W)))   # Clip max=W
        y1 = int(np.round(np.clip(r_y - r_h / 2, 0, None)))   # Clip min=0
        y2 = int(np.round(np.clip(r_y + r_h / 2, None, H)))   # Clip max=H

        inputs[:, :, y1:y2, x1:x2] = shuffled_inputs[:, :, y1:y2, x1:x2]
        adjusted_lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        mixed_labels = adjusted_lam * targets + (1 - adjusted_lam) * shuffled_targets
        
        return inputs, mixed_labels