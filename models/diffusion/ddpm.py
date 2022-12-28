
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseGenerativeModel
from .unet import UNet

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DDPM(BaseGenerativeModel):
    def __init__(self, cfg):
        super().__init__()
        
        if cfg["model"]["backbone"]["name"] == "unet":
            self.model = UNet(**cfg["model"]["backbone"]["kwargs"])
        
        self.T = cfg["model"]["T"]
        self.beta_1 = cfg["model"]["beta_1"]
        self.beta_T = cfg["model"]["beta_T"]
        self.img_shape = cfg["model"]["img_shape"]
        
        self.register_buffer(
            'betas', torch.linspace(self.beta_1, self.beta_T, self.T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))


    def forward_loss(self, x):
        """
        Algorithm 1.
        """
        x_0 = x
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return F.mse_loss(self.model(x_t, t), noise, reduction='sum') / self.T

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def sample(self, num_samples):
        """
        Algorithm 2.
        """
        x_t = torch.randn(
            size=[num_samples, *self.img_shape]).cuda()
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([num_samples, ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return x_0
