from torch import nn
from abc import abstractmethod

class BaseGenerativeModel(nn.Module):
    
    def __init__(self) -> None:
        super(BaseGenerativeModel, self).__init__()


    def sample(self, num_samples):
        raise NotImplementedError

    def forward(self, x=None, num_samples=None, **kwargs):
        if x is not None:
            return self.forward_loss(x, **kwargs)
        else:
            return self.sample(num_samples, **kwargs)

    @abstractmethod
    def forward_loss(self, x):
        pass



