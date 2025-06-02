import torch

@torch.no_grad()
def sample_interpolation(model, p1, p2, device, N=15):
    """ Sampling N points from the line that connects p1 and p2 """
    
    weights = torch.linspace(0, 1, N).view(15, 1).to(device)
    z = p1 * (1 - weights) + p2 * weights
    decoded_conv = model.decoder_fc(z)
    x_dec = decoded_conv.view(N, 128, 8, 8)
    decoded_imgs = model.decoder_deconv(x_dec).cpu()
    
    return decoded_imgs
