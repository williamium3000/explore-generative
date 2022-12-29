from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from .gan import GAN

def loss_fn_g(gen_x_d):
    return -torch.mean(gen_x_d)

def loss_fn_d(gen_x_d, x_d):
    return -torch.mean(x_d) + torch.mean(gen_x_d)


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        return self.model(img)

class WGAN(GAN):
    def __init__(self, cfg) -> None:
        super(WGAN, self).__init__(cfg)
        self.loss_fn_g = loss_fn_g
        self.loss_fn_d = loss_fn_d
        self.D = Discriminator(self.img_shape)


    