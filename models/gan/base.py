from torch import nn
import torch
from ..base import BaseGenerativeModel

class BaseGanModel(BaseGenerativeModel):
    
    def __init__(self) -> None:
        super(BaseGanModel, self).__init__()
        self.G = None
        self.D = None
        self.loss_fn_g = None
        self.loss_fn_d = None

    def forward_loss(self, x):
        z = torch.randn(x.size(0),
                        self.latent_dim).cuda()
        gen_x = self.G(z)

        return self.loss_fn_g(self.D(gen_x)), self.loss_fn_d(self.D(gen_x.detach()), self.D(x))
    
    def sample(self, num_samples):
        z = torch.randn(num_samples,
                        self.latent_dim).cuda()
        return self.G(z)

        
    