import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)
    return activation()

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder
    
    Use Conv layers for encoder,
    ConvTranspose layers for decoder, and Linear layers only for mu and sigma
    """
    def __init__(self, in_size=(3, 64, 64), latent_dim=32, act="ReLU"):
        super().__init__()
        self.in_size = in_size
        self.latent_dim = latent_dim
        self.activation = get_act(act)
        C, _, _ = in_size

        # Encoder (Conv2d blocks -> flatten -> fc_mu/sigma)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),  # -> (32, H/2, W/2)
            self.activation,
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, H/4, W/4)
            self.activation,
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, H/8, W/8)
            self.activation
        )

        # Compute conv output size for flattening
        conv_out = self.encoder_conv(torch.zeros(1, *in_size))
        self.flat_dim = conv_out.view(1, -1).shape[1]

        # Latent variables
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_sigma = nn.Linear(self.flat_dim, latent_dim)

        # Decoder (fc -> unflatten -> ConvTranspose2d)
        self.decoder_fc = nn.Linear(latent_dim, self.flat_dim)

        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, H/4, W/4)
            self.activation,
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (32, H/2, W/2)
            self.activation,
            nn.ConvTranspose2d(32, C, kernel_size=4, stride=2, padding=1),    # -> (C, H, W)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        batch_size = x.size(0)

        x_enc = self.encoder_conv(x)
        x_flat = x_enc.view(batch_size, -1)

        mu = self.fc_mu(x_flat)
        log_var = self.fc_sigma(x_flat)
        z = self.reparameterize(mu, log_var)

        x_dec_flat = self.decoder_fc(z)
        x_dec = x_dec_flat.view(batch_size, 128, x_enc.size(2), x_enc.size(3))
        x_hat = self.decoder_deconv(x_dec)

        return x_hat, (z, mu, log_var)