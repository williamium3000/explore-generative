from diffusion.ddpm import DDPM
from vae.vae_mlp import VAE as VAE_MLP
from vae.vae_cnn import VAE as VAE_CNN

def build_model(cfg):
    if cfg["model"]["name"] == "ddpm":
        model_fn = DDPM
    elif cfg["model"]["name"] == "vae_mlp":
        model_fn = VAE_MLP
    elif cfg["model"]["name"] == "vae_cnn":
        model_fn = VAE_CNN
    return model_fn(cfg)