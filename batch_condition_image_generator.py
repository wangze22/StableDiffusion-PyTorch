"""
Batch generation script that iterates over every available condition pair
in the configured dataset and produces multiple samples per pair.

The sampling logic mirrors the GUI workflow in ``Model_DiT_9L_GUI.py`` but
removes all interactive components so that large batches can be rendered
offline. All configurable knobs live in the block under
``if __name__ == '__main__':`` for convenience.
"""

from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from models.transformer import DIT
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import DDIMSampler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from cim_qn_train.progressive_qn_train import ProgressiveTrain
import cim_layers.register_dict as reg_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed: Optional[int]) -> None:
    """Set all relevant RNG seeds so that identical inputs reproduce results."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _latent_size(cfg) -> int:
    downsample_count = sum(int(flag) for flag in cfg.autoencoder_down_sample)
    return cfg.dataset_im_size // (2 ** downsample_count)


def load_text_components(cfg):
    if 'text' not in cfg.ldm_condition_types:
        return None, None, None
    with torch.no_grad():
        tokenizer, model = get_tokenizer_and_model(
            cfg.ldm_text_condition_text_embed_model,
            device = device,
        )
        empty_embed = get_text_representation([''], tokenizer, model, device)
    return tokenizer, model, empty_embed


def build_dit_model(cfg, ldm_ckpt_path: Path) -> torch.nn.Module:
    if not ldm_ckpt_path.exists():
        raise FileNotFoundError(f'Could not find diffusion checkpoint: {ldm_ckpt_path}')
    model = DIT(
        im_channels = cfg.autoencoder_z_channels,
        model_config = cfg.dit_model_config,
    ).to(device)
    trainer = ProgressiveTrain(model)
    trainer.convert_to_layers(
        convert_layer_type_list = reg_dict.nn_layers,
        tar_layer_type = 'layers_qn_lsq',
        noise_scale = 0.05,
        input_bit = 8,
        output_bit = 8,
        weight_bit = 4,
    )
    trainer.add_enhance_branch_LoR(ops_factor = 0.05)
    trainer.add_enhance_layers(ops_factor = 0.05)
    state_dict = torch.load(str(ldm_ckpt_path), map_location = device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_vae(cfg, vqvae_ckpt_path: Path) -> VQVAE:
    if not vqvae_ckpt_path.exists():
        raise FileNotFoundError(f'Could not find VQVAE checkpoint: {vqvae_ckpt_path}')
    autoencoder_config = {
        'z_channels'     : cfg.autoencoder_z_channels,
        'codebook_size'  : cfg.autoencoder_codebook_size,
        'down_channels'  : list(cfg.autoencoder_down_channels),
        'mid_channels'   : list(cfg.autoencoder_mid_channels),
        'down_sample'    : list(cfg.autoencoder_down_sample),
        'attn_down'      : list(cfg.autoencoder_attn_down),
        'norm_channels'  : cfg.autoencoder_norm_channels,
        'num_heads'      : cfg.autoencoder_num_heads,
        'num_down_layers': cfg.autoencoder_num_down_layers,
        'num_mid_layers' : cfg.autoencoder_num_mid_layers,
        'num_up_layers'  : cfg.autoencoder_num_up_layers,
    }
    vae = VQVAE(im_channels = cfg.dataset_im_channels, model_config = autoencoder_config).to(device)
    vae.eval()
    vae.load_state_dict(torch.load(str(vqvae_ckpt_path), map_location = device))
    return vae


def build_dataset(cfg) -> CelebDataset:
    if cfg.condition_config is None:
        raise ValueError('condition_config must be defined inside the config module.')
    dataset = CelebDataset(
        split = 'train',
        im_path = cfg.dataset_im_path,
        im_size = cfg.dataset_im_size,
        im_channels = cfg.dataset_im_channels,
        use_latents = False,
        latent_path = None,
        condition_config = cfg.condition_config,
    )
    return dataset


def read_first_caption(path: str) -> str:
    with open(path, 'r', encoding = 'utf-8') as fp:
        for line in fp:
            stripped = line.strip()
            if stripped:
                return stripped
    return ''


@dataclass
class ConditionPayload:
    index: int
    image_id: str
    mask: Optional[torch.Tensor]
    prompt_embed: Optional[torch.Tensor]


def gather_condition_payloads(
    dataset: CelebDataset,
    cond_types: Iterable[str],
    text_tokenizer,
    text_model,
    max_items: Optional[int],
) -> Iterable[ConditionPayload]:
    text_enabled = 'text' in cond_types
    image_enabled = 'image' in cond_types
    total = len(dataset) if max_items is None else min(len(dataset), max_items)

    for idx in range(total):
        image_path = Path(dataset.images[idx])
        prompt = None
        prompt_embed = None
        mask = None

        if text_enabled:
            caption_path = dataset.texts[idx]
            prompt = read_first_caption(caption_path)
            with torch.no_grad():
                prompt_embed = get_text_representation([prompt], text_tokenizer, text_model, device)

        if image_enabled:
            mask_tensor = dataset.get_mask(idx).float()
            mask = mask_tensor.unsqueeze(0).to(device)

        yield ConditionPayload(
            index = idx,
            image_id = image_path.stem,
            mask = mask,
            prompt_embed = prompt_embed,
        )


def prepare_condition_inputs(
    cond_types: Iterable[str],
    payload: ConditionPayload,
    empty_text_embed: Optional[torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    cond_input: Dict[str, torch.Tensor] = {}
    uncond_input: Dict[str, torch.Tensor] = {}

    if 'image' in cond_types:
        assert payload.mask is not None, 'Image conditioning requested but mask missing.'
        cond_input['image'] = payload.mask
        uncond_input['image'] = torch.zeros_like(payload.mask)

    if 'text' in cond_types:
        assert payload.prompt_embed is not None, 'Text conditioning requested but prompt missing.'
        assert empty_text_embed is not None, 'Empty text embedding must be precomputed.'
        cond_input['text'] = payload.prompt_embed
        uncond_input['text'] = empty_text_embed

    return cond_input, uncond_input


def sample_image(
    sampler: DDIMSampler,
    vae: VQVAE,
    cond_input: Dict[str, torch.Tensor],
    uncond_input: Dict[str, torch.Tensor],
    latent_size: int,
    latent_channels: int,
    num_inference_steps: int,
    guidance_scale: float,
    method: str,
    eta: float,
) -> Image.Image:
    xt = torch.randn(
        (1, latent_channels, latent_size, latent_size),
        device = device,
    )

    class _GuidedWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x_t, t, cond):
            scale = max(0.0, float(guidance_scale))
            if scale == 0.0:
                return self.base_model(x_t, t, uncond_input)
            cond_pred = self.base_model(x_t, t, cond)
            if abs(scale - 1.0) < 1e-6:
                return cond_pred
            uncond_pred = self.base_model(x_t, t, uncond_input)
            return uncond_pred + scale * (cond_pred - uncond_pred)

    original_model = sampler.model
    sampler.model = _GuidedWrapper(original_model)
    try:
        with torch.no_grad():
            latents = sampler.forward(
                xt,
                cond_input,
                uncond_input,
                steps = num_inference_steps,
                method = method,
                eta = eta,
            )
            decoded = vae.decode(latents)
            decoded = torch.clamp(decoded, -1.0, 1.0).detach().cpu()
            decoded = (decoded + 1) / 2
            grid = make_grid(decoded, nrow = 1)
            img = torchvision.transforms.ToPILImage()(grid)
        return img
    finally:
        sampler.model = original_model


def run_generation(
    cfg,
    vqvae_ckpt: Path,
    ldm_ckpt: Path,
    output_dir: Path,
    samples_per_condition: int,
    guidance_scale: float,
    num_inference_steps: int,
    sampler_method: str,
    sampler_eta: float,
    skip_existing: bool,
    limit_num_items: Optional[int],
    seed: Optional[int],
) -> None:
    if samples_per_condition < 1:
        raise ValueError('samples_per_condition must be >= 1')

    seed_everything(seed)

    condition_types = cfg.ldm_condition_types
    if not condition_types:
        raise ValueError('At least one condition type must be enabled in the config.')

    text_tokenizer, text_model, empty_text_embed = load_text_components(cfg)

    model = build_dit_model(cfg, ldm_ckpt)
    vae = build_vae(cfg, vqvae_ckpt)
    sampler = DDIMSampler(
        model = model,
        beta = (cfg.diffusion_beta_start, cfg.diffusion_beta_end),
        T = cfg.diffusion_num_timesteps,
    )

    dataset = build_dataset(cfg)
    total_items = len(dataset)
    if limit_num_items is not None:
        total_items = min(total_items, limit_num_items)

    output_dir.mkdir(parents = True, exist_ok = True)
    latent_size = _latent_size(cfg)

    payload_iter = gather_condition_payloads(
        dataset = dataset,
        cond_types = condition_types,
        text_tokenizer = text_tokenizer,
        text_model = text_model,
        max_items = limit_num_items,
    )

    generated = 0
    for payload in tqdm(payload_iter, total = total_items, desc = 'Sampling'):

        cond_input, uncond_input = prepare_condition_inputs(
            condition_types,
            payload,
            empty_text_embed,
        )

        sub_dir = output_dir / payload.image_id
        sub_dir.mkdir(parents = True, exist_ok = True)

        for sample_idx in range(samples_per_condition):
            save_name = f'{payload.image_id}_s{sample_idx:02d}.png'
            output_path = sub_dir / save_name
            if skip_existing and output_path.exists():
                continue
            image = sample_image(
                sampler = sampler,
                vae = vae,
                cond_input = cond_input,
                uncond_input = uncond_input,
                latent_size = latent_size,
                latent_channels = cfg.autoencoder_z_channels,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                method = sampler_method,
                eta = sampler_eta,
            )
            image.save(output_path)
            generated += 1

    print(f'Finished sampling {generated} images into {output_dir}')


if __name__ == '__main__':
    # ------------------------------------------------------------------- #
    # All user-facing knobs live here. Update the strings/values directly.
    CONFIG_MODULE = 'Model_DiT_9L_config'
    VQVAE_CKPT = 'runs_VQVAE_noise_server/vqvae_20251028-131331/celebhq/n_scale_0.2000/vqvae_autoencoder_ckpt_latest.pth'
    LDM_CKPT = 'runs_DiT_9L_server/ddpm_20251105-231756_save/LSQ_AnDi/w4b_0.098776/ddpm_ckpt_text_image_cond_clip.pth'
    OUTPUT_DIR = 'generated/batch_conditions'

    SAMPLES_PER_CONDITION = 2
    GUIDANCE_SCALE = 1.0
    NUM_INFERENCE_STEPS = 20
    SAMPLER_METHOD = 'quadratic'  # 'linear' or 'quadratic'
    SAMPLER_ETA = 1.0  # 0 -> DDIM, 1 -> DDPM
    SKIP_EXISTING = True
    LIMIT_NUM_ITEMS = None  # e.g. 100 for quick smoke test
    GLOBAL_SEED = 12345
    # ------------------------------------------------------------------- #

    cfg_module = importlib.import_module(CONFIG_MODULE)
    run_generation(
        cfg = cfg_module,
        vqvae_ckpt = Path(VQVAE_CKPT),
        ldm_ckpt = Path(LDM_CKPT),
        output_dir = Path(OUTPUT_DIR),
        samples_per_condition = SAMPLES_PER_CONDITION,
        guidance_scale = GUIDANCE_SCALE,
        num_inference_steps = NUM_INFERENCE_STEPS,
        sampler_method = SAMPLER_METHOD,
        sampler_eta = SAMPLER_ETA,
        skip_existing = SKIP_EXISTING,
        limit_num_items = LIMIT_NUM_ITEMS,
        seed = GLOBAL_SEED,
    )
