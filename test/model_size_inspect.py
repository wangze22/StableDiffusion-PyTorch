import sys
from pathlib import Path

import yaml
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configure script behaviour here.
CONFIG_PATH = 'config/celebhq.yaml'
INCLUDE_CLIP = True
# Optional: local directory or HuggingFace repo id for CLIP weights.
CLIP_MODEL_PATH = None

from models.vqvae import VQVAE
from models.unet_base import Unet as UNet


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_modules(*modules: torch.nn.Module) -> int:
    total = 0
    for m in modules:
        if m is None:
            continue
        total += count_params(m)
    return total


def load_config(path: str):
    cfg_path = Path(path)
    if not cfg_path.is_file():
        cfg_path = ROOT_DIR / path
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fmt_millions(n_params: int) -> str:
    return f"{n_params / 1e6:.3f} M"


def build_vqvae(cfg):
    im_channels = cfg['dataset_params']['im_channels']
    vq_cfg = cfg['autoencoder_params']
    return VQVAE(im_channels=im_channels, model_config=vq_cfg)


def build_unet(cfg):
    # Diffusion UNet works on latents with z_channels channels
    z_channels = cfg['autoencoder_params']['z_channels']
    unet_cfg = cfg['ldm_params']
    return UNet(im_channels=z_channels, model_config=unet_cfg)


def build_clip_text_model(model_path: str | None = None):
    """Load CLIP text encoder from HuggingFace or a local directory."""
    from transformers import CLIPTextModel

    if model_path is None:
        model_path = 'openai/clip-vit-base-patch16'
    return CLIPTextModel.from_pretrained(model_path)


def main():
    cfg = load_config(CONFIG_PATH)

    # VQVAE (full)
    vqvae = build_vqvae(cfg)
    vq_total = count_params(vqvae)

    # VQVAE decoder-only
    dec_params = count_modules(
        getattr(vqvae, 'post_quant_conv', None),
        getattr(vqvae, 'decoder_conv_in', None),
        getattr(vqvae, 'decoder_mids', None),
        getattr(vqvae, 'decoder_layers', None),
        getattr(vqvae, 'decoder_norm_out', None),
        getattr(vqvae, 'decoder_conv_out', None),
    )

    # Diffusion UNet
    unet = build_unet(cfg)
    unet_total = count_params(unet)

    # CLIP text encoder (optional)
    clip_total = None
    if INCLUDE_CLIP:
        try:
            clip = build_clip_text_model(CLIP_MODEL_PATH)
            clip_total = count_params(clip)
        except OSError as err:
            print(f"Warning: CLIP model unavailable ({err}). Set INCLUDE_CLIP=False or point CLIP_MODEL_PATH to a local cache.")

    print('Model Parameter Sizes (Millions):')
    print(f"- VQVAE (full):        {fmt_millions(vq_total)}")
    print(f"- VQVAE Decoder only:  {fmt_millions(dec_params)}")
    print(f"- Diffusion UNet:      {fmt_millions(unet_total)}")
    if clip_total is not None:
        print(f"- CLIP Text Model:     {fmt_millions(clip_total)}")
    else:
        print("- CLIP Text Model:     [skipped] (set INCLUDE_CLIP=True and ensure weights are available)")


if __name__ == '__main__':
    main()
