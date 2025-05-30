import torch

@torch.no_grad()
def encode_samples(model, images, device):
    encoded1 = model.encoder_conv(images[0].to(device)).view(1, -1)
    mean1 = model.fc_mu(encoded1)
    log_var1 = model.fc_sigma(encoded1)
    z1 = model.reparameterize(mean1, log_var1)

    encoded2 = model.encoder_conv(images[1].to(device)).view(1, -1)
    mean2 = model.fc_mu(encoded2)
    log_var2 = model.fc_sigma(encoded2)
    z2 = model.reparameterize(mean2, log_var2)
    
    return z1, z2
