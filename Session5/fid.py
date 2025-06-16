import torch
from torchvision.models import inception_v3
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision import transforms


def create_samples(generator, num_samples, device):
    z = torch.randn(num_samples, 128, 1, 1).to(device)
    fake_img = generator(z)
    
    return fake_img.to(device)


@torch.no_grad()
def compute_fid(generator, dataloader, device, num_samples = 1000):
    """" 
    The Frechet distance between two Gaussians:
    FID = ||mean_1 - mean_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    # since inception requires 299x299 normalized images
    inception_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.fc = nn.Identity()
    
    real_features = []
    fake_features = []
    
    for i, (data, _) in enumerate(dataloader):
        if i * dataloader.batch_size >= num_samples:
            break
        data = inception_transform(data.to(device))
        features = inception(data)
        real_features.append(features.cpu())

    samples = create_samples(generator, num_samples, device)
    features = inception(inception_transform(samples))
    fake_features.append(features.cpu())
    
    real_features = torch.cat(real_features, dim=0).numpy()
    fake_features = torch.cat(fake_features, dim=0).numpy()
    
    mean_real, cov_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mean_fake, cov_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    covmean = sqrtm(cov_real.dot(cov_fake))
    
    diff = mean_real - mean_fake
    fid = diff.dot(diff) + np.trace(cov_real + cov_fake - 2 * covmean)

    del data, samples, features, fake_features, real_features
    torch.cuda.empty_cache()
    
    return fid
