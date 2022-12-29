from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .base import BaseGanModel

def loss_fn_g(x_d):
    return F.binary_cross_entropy(x_d, torch.ones((x_d.size(0), 1)).cuda())

def loss_fn_d(gen_x_d, x_d):
    return (F.binary_cross_entropy(gen_x_d, torch.zeros((x_d.size(0), 1)).cuda()) + F.binary_cross_entropy(x_d, torch.ones((x_d.size(0), 1)).cuda())) / 2
    
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)


class GAN(BaseGanModel):
    
    def __init__(self, cfg) -> None:
        super(GAN, self).__init__()
        self.latent_dim = cfg["model"]["latent_dim"]
        self.img_shape = cfg["model"]["img_shape"]
        self.G = Generator(self.latent_dim, self.img_shape)
        self.D = Discriminator(self.img_shape)
        self.loss_fn_g = loss_fn_g
        self.loss_fn_d = loss_fn_d


    