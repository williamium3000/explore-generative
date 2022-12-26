import torch
import torch.nn as nn

def bce_kl_loss(gen_x, x, mu, logvar):
    """
    gen_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = nn.BCELoss(reduction="sum")(gen_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # print(KLD_element.shape)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return (BCE +  KLD) / x.size(0)

def mse_kl_loss(gen_x, x, mu, logvar):
    """
    gen_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = nn.MSELoss(reduction="sum")(gen_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # print(KLD_element.shape)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return (MSE +  KLD) / x.size(0)

class VAE(nn.Module):
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
            nn.Sigmoid()
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

    def forward(self, x=None, num_samples=None):
        if x is not None:
            return self.forward_loss(x)
        else:
            return self.sample(num_samples)
    
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

