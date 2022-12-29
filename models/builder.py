from .diffusion.ddpm import DDPM
from .vae.vae_mlp import VAE as VAE_MLP
from .vae.vae_cnn import VAE as VAE_CNN
from .gan.gan import GAN
from .gan.dcgan import DCGAN
from .gan.wgan import WGAN

def build_model(cfg):
    if cfg["model"]["name"] == "ddpm":
        model_fn = DDPM
    elif cfg["model"]["name"] == "vae_mlp":
        model_fn = VAE_MLP
    elif cfg["model"]["name"] == "vae_cnn":
        model_fn = VAE_CNN
    elif cfg["model"]["name"] == "gan":
        model_fn = GAN
    elif cfg["model"]["name"] == "dcgan":
        model_fn = DCGAN
    elif cfg["model"]["name"] == "wgan":
        model_fn = WGAN
    return model_fn(cfg)