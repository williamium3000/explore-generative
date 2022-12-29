from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .gan import GAN

class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim=128, out_channel=3):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, latent_dim * 8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(latent_dim * 8)
        self.deconv2 = nn.ConvTranspose2d(latent_dim * 8, latent_dim * 4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(latent_dim * 4)
        self.deconv3 = nn.ConvTranspose2d(latent_dim * 4, latent_dim * 2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(latent_dim * 2)
        self.deconv4 = nn.ConvTranspose2d(latent_dim * 2, latent_dim, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(latent_dim)
        self.deconv5 = nn.ConvTranspose2d(latent_dim, out_channel, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input.view(input.size(0), -1, 1, 1))))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

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
        x = torch.sigmoid(self.conv5(x)).view(input.size(0), 1)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DCGAN(GAN):
    
    def __init__(self, cfg) -> None:
        super(DCGAN, self).__init__(cfg)
        self.latent_dim = cfg["model"]["latent_dim"]
        self.img_shape = cfg["model"]["img_shape"]
        self.G = Generator(self.latent_dim, self.img_shape[0])
        self.D = Discriminator(self.latent_dim, self.img_shape[0])


    