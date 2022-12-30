from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .dcgan import DCGAN, normal_init

def loss_fn_g(gen_x_d):
    return -torch.mean(gen_x_d)

def loss_fn_d(gen_x_d, x_d):
    return -torch.mean(x_d) + torch.mean(gen_x_d)


class Discriminator(nn.Module):
    # initializers
    def __init__(self, latent_dim=128, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, latent_dim, 4, 2, 1)
        self.conv2 = nn.Conv2d(latent_dim, latent_dim * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(latent_dim * 2)
        self.conv3 = nn.Conv2d(latent_dim * 2, latent_dim * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(latent_dim * 4)
        self.conv4 = nn.Conv2d(latent_dim * 4, latent_dim * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(latent_dim * 8)
        self.conv5 = nn.Conv2d(latent_dim * 8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x).view(input.size(0), 1)

        return x


class WGAN(DCGAN):
    def __init__(self, cfg) -> None:
        super(WGAN, self).__init__(cfg)
        self.loss_fn_g = loss_fn_g
        self.loss_fn_d = loss_fn_d
        
        self.latent_dim = cfg["model"]["latent_dim"]
        self.img_shape = cfg["model"]["img_shape"]
        self.D = Discriminator(self.latent_dim, self.img_shape[0])
        self.D.weight_init(mean=0.0, std=0.02)


    