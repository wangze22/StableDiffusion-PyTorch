import os
import sys
import yaml
import torch

from models.vqvae import VQVAE
from models.unet_cond_base import Unet as UNet


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
    with open(path, 'r', encoding='utf-8') as f:
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
    import argparse

    parser = argparse.ArgumentParser(description='Inspect parameter sizes (in M) of core models')
    parser.add_argument('--config', default='config/celebhq_text_image_cond.yaml', type=str,
                        help='Path to YAML config with VQVAE/UNet params')
    parser.add_argument('--no-clip', action='store_true', help='Skip CLIP model (avoids download)')
    parser.add_argument('--clip-path', type=str, default=None,
                        help='Optional local directory or HF repo id for CLIP text model')
    args = parser.parse_args()

    cfg = load_config(args.config)

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
    if not args.no_clip:
        try:
            clip = build_clip_text_model(args.clip_path)
            clip_total = count_params(clip)
        except OSError as err:
            print(f"Warning: CLIP model unavailable ({err}). Use --clip-path or --no-clip to skip.")

    print('Model Parameter Sizes (Millions):')
    print(f"- VQVAE (full):        {fmt_millions(vq_total)}")
    print(f"- VQVAE Decoder only:  {fmt_millions(dec_params)}")
    print(f"- Diffusion UNet:      {fmt_millions(unet_total)}")
    if clip_total is not None:
        print(f"- CLIP Text Model:     {fmt_millions(clip_total)}")
    else:
        print("- CLIP Text Model:     [skipped] (use without --no-clip to include)")


if __name__ == '__main__':
    main()
