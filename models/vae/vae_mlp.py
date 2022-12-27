import torch
from torch.nn import functional as F
import torch.nn as nn

from ..base import BaseGenerativeModel
def bce_kl_loss(gen_x, x, mu, logvar):
    """
    gen_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    bce_loss = nn.BCELoss(reduction="sum")(gen_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).mean().mul_(-0.5) # 512
    return bce_loss +  kld_loss / 6

def mse_kl_loss(gen_x, x, mu, logvar):
    """
    gen_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse_loss = F.mse_loss(gen_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).mean().mul_(-0.5)
    return mse_loss +  kld_loss / 6

class VAE(BaseGenerativeModel):
    def __init__(self, input_size, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(512, self.latent_dim)
        self.log_std_head = nn.Linear(512, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Tanh()
        )


    def encode(self, x):
        feature = self.backbone(x)
        return self.mean_head(feature), self.log_std_head(feature)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)
    
    def forward_loss(self, x):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return mse_kl_loss(self.decode(z), x, mu, logvar)

    
    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim).cuda()
        samples = self.decode(z)
        return samples

