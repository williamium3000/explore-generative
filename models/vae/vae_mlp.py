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
    kld_loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).mean().mul_(-0.5)
    return (bce_loss +  kld_loss) / x.size(0)

def mse_kl_loss(gen_x, x, mu, logvar):
    """
    gen_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse_loss = nn.MSELoss(reduction="sum")(gen_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).sum().mul_(-0.5)
    return (mse_loss +  kld_loss) / x.size(0)

class VAE(BaseGenerativeModel):
    def __init__(self, cfg):
        super(VAE, self).__init__()
        
        self.input_size = cfg["model"]["input_size"]
        self.latent_dim = cfg["model"]["latent_dim"]
        self.img_shape = cfg["model"]["img_shape"]
        self.loss_fn = mse_kl_loss if cfg["model"]["loss"] == "mse_kl_loss" else bce_kl_loss
        
        self.backbone = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(512, self.latent_dim)
        self.log_std_head = nn.Linear(512, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_size),
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
        return self.loss_fn(self.decode(z), x, mu, logvar)

    
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
        return samples.reshape(num_samples, *self.img_shape)

