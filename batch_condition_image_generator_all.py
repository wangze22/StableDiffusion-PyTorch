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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from models.transformer import DIT
from models.unet_cond_base_relu import Unet

from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import DDIMSampler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from cim_qn_train.progressive_qn_train import ProgressiveTrain
import cim_layers.register_dict as reg_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
available_gpu_count = torch.cuda.device_count() if device.type == 'cuda' else 0
to_pil = ToPILImage()


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


def build_dit_model(cfg, ldm_ckpt_path: Path, use_data_parallel: bool) -> torch.nn.Module:
    if not ldm_ckpt_path.exists():
        raise FileNotFoundError(f'Could not find diffusion checkpoint: {ldm_ckpt_path}')
    cfg_name = getattr(cfg, '__name__', '')
    if cfg_name == 'Model_Unet_config':
        model = Unet(
            im_channels = cfg.autoencoder_z_channels,
            model_config = cfg.diffusion_model_config,
        ).to(device)
    else:
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
    if use_data_parallel and available_gpu_count > 1:
        model = torch.nn.DataParallel(model)
        model.eval()
    return model


def build_vae(cfg, vqvae_ckpt_path: Path, use_data_parallel: bool) -> VQVAE:
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
    vae.load_state_dict(torch.load(str(vqvae_ckpt_path), map_location = device))
    vae.eval()
    if use_data_parallel and available_gpu_count > 1:
        vae = torch.nn.DataParallel(vae)
        vae.eval()
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


def sample_batch(
    sampler: DDIMSampler,
    vae: VQVAE,
    cond_stack: Dict[str, List[torch.Tensor]],
    uncond_stack: Dict[str, List[torch.Tensor]],
    latent_size: int,
    latent_channels: int,
    num_inference_steps: int,
    guidance_scale: float,
    method: str,
    eta: float,
    batch_size: int,
) -> List[Image.Image]:
    cond_input = {key: torch.cat(tensors, dim = 0).to(device) for key, tensors in cond_stack.items()}
    uncond_input = {key: torch.cat(tensors, dim = 0).to(device) for key, tensors in uncond_stack.items()}

    xt = torch.randn(
        (batch_size, latent_channels, latent_size, latent_size),
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
            vae_module = vae.module if isinstance(vae, torch.nn.DataParallel) else vae
            decoded = vae_module.decode(latents)
            decoded = torch.clamp(decoded, -1.0, 1.0).detach().cpu()
            decoded = (decoded + 1) / 2
            images = [to_pil(sample) for sample in decoded]
        return images
    finally:
        sampler.model = original_model


def run_generation(
    cfg,
    vqvae_ckpt: Path,
    ldm_ckpt: Path,
    output_dir: Path,
    batch_size: int,
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
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')

    active_gpus = available_gpu_count
    device_multiplier = max(1, active_gpus)
    effective_batch_size = batch_size * device_multiplier

    if effective_batch_size % samples_per_condition != 0:
        raise ValueError('batch_size * num_devices must be divisible by samples_per_condition for efficient batching.')

    if active_gpus > 1:
        print(f'Detected {active_gpus} GPUs. Each GPU will sample {batch_size} images '
              f'per iteration (effective batch size = {effective_batch_size}).')

    seed_everything(seed)
    conditions_per_batch = effective_batch_size // samples_per_condition

    condition_types = cfg.ldm_condition_types
    if not condition_types:
        raise ValueError('At least one condition type must be enabled in the config.')

    text_tokenizer, text_model, empty_text_embed = load_text_components(cfg)

    use_data_parallel = active_gpus > 1
    model = build_dit_model(cfg, ldm_ckpt, use_data_parallel = use_data_parallel)
    vae = build_vae(cfg, vqvae_ckpt, use_data_parallel = use_data_parallel)
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

    # Write sampling configuration into OUTPUT_DIR as YAML
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # Fallback to manual writer if PyYAML is unavailable

    config_data = {
        'CONFIG_MODULE': getattr(cfg, '__name__', str(cfg)),
        'VQVAE_CKPT': str(vqvae_ckpt),
        'LDM_CKPT': str(ldm_ckpt),
        'OUTPUT_DIR': str(output_dir),
        'BATCH_SIZE': batch_size,
        'SAMPLES_PER_CONDITION': samples_per_condition,
        'GUIDANCE_SCALE': guidance_scale,
        'NUM_INFERENCE_STEPS': num_inference_steps,
        'SAMPLER_METHOD': sampler_method,
        'SAMPLER_ETA': sampler_eta,
        'SKIP_EXISTING': skip_existing,
        'LIMIT_NUM_ITEMS': limit_num_items,
        'GLOBAL_SEED': seed,
    }
    config_path = output_dir / 'sampling_config.yaml'
    if yaml is not None:
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, sort_keys = False, allow_unicode = True)
        except Exception:
            yaml = None  # Fall back to manual writer below

    if yaml is None:
        def _to_yaml_literal(val):
            if isinstance(val, bool):
                return 'true' if val else 'false'
            if val is None:
                return 'null'
            if isinstance(val, (int, float)):
                return str(val)
            s = str(val).replace("'", "''")
            return f"'{s}'"

        with open(config_path, 'w', encoding='utf-8') as f:
            for k, v in config_data.items():
                f.write(f"{k}: {_to_yaml_literal(v)}\n")

    latent_size = _latent_size(cfg)

    payload_iter = gather_condition_payloads(
        dataset = dataset,
        cond_types = condition_types,
        text_tokenizer = text_tokenizer,
        text_model = text_model,
        max_items = limit_num_items,
    )

    def process_batch(batch_payloads: List[ConditionPayload]) -> int:
        cond_stack: Dict[str, List[torch.Tensor]] = defaultdict(list)
        uncond_stack: Dict[str, List[torch.Tensor]] = defaultdict(list)
        metadata: List[Tuple[Path, int, str, int]] = []

        for payload in batch_payloads:
            cond_input, uncond_input = prepare_condition_inputs(
                condition_types,
                payload,
                empty_text_embed,
            )
            for sample_idx in range(samples_per_condition):
                save_name = f'{payload.index:06d}_{payload.image_id}_s{sample_idx:02d}.png'
                output_path = output_dir / save_name
                if skip_existing and output_path.exists():
                    continue
                for key, tensor in cond_input.items():
                    cond_stack[key].append(tensor)
                for key, tensor in uncond_input.items():
                    uncond_stack[key].append(tensor)
                metadata.append((output_path, payload.index, payload.image_id, sample_idx))

        if not metadata:
            return 0

        images = sample_batch(
            sampler = sampler,
            vae = vae,
            cond_stack = cond_stack,
            uncond_stack = uncond_stack,
            latent_size = latent_size,
            latent_channels = cfg.autoencoder_z_channels,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            method = sampler_method,
            eta = sampler_eta,
            batch_size = len(metadata),
        )
        for img, (path, _, _, _) in zip(images, metadata):
            img.save(path)
        return len(metadata)

    generated = 0
    batch_payloads: List[ConditionPayload] = []
    for payload in tqdm(payload_iter, total = total_items, desc = 'Sampling'):
        batch_payloads.append(payload)
        if len(batch_payloads) == conditions_per_batch:
            generated += process_batch(batch_payloads)
            batch_payloads.clear()
    if batch_payloads:
        generated += process_batch(batch_payloads)

    print(f'Finished sampling {generated} images into {output_dir}')


if __name__ == '__main__':
    # ------------------------------------------------------------------- #
    # All user-facing knobs live here. Update the strings/values directly.
    VQVAE_CKPT = 'runs_VQVAE_noise_server/vqvae_20251028-131331/celebhq/n_scale_0.2000/vqvae_autoencoder_ckpt_latest.pth'
    RUN_CONFIGS = [
        {
            'config_module': 'Model_DiT_9L_config',
            'ldm_ckpt': 'runs_DiT_9L_server/ddpm_20251105-231756_save/LSQ_AnDi/w4b_0.098776/ddpm_ckpt_text_image_cond_clip.pth',
            'output_dir': 'FID_Images/DiT_9L',
        },
        {
            'config_module': 'Model_DiT_12L_config',
            'ldm_ckpt': 'runs_DiT_12L_server/ddpm_20251103-232943_save/LSQ_AnDi/w4b_0.097959/ddpm_ckpt_text_image_cond_clip.pth',
            'output_dir': 'FID_Images/DiT_12L',
        },
        {
            'config_module': 'Model_Unet_config',
            'ldm_ckpt': 'runs_Unet_server/ddpm_20251104-133643/LSQ_AnDi/w4b_0.087755/ddpm_ckpt_text_image_cond_clip.pth',
            'output_dir': 'FID_Images/Unet',
        },
    ]

    BATCH_SIZE = 32  # Per GPU; total batch grows with the number of GPUs.
    SAMPLES_PER_CONDITION = 2
    GUIDANCE_SCALE = 1.0
    NUM_INFERENCE_STEPS = 20
    SAMPLER_METHOD = 'quadratic'  # 'linear' or 'quadratic'
    SAMPLER_ETA = 1.0  # 0 -> DDIM, 1 -> DDPM
    SKIP_EXISTING = True
    LIMIT_NUM_ITEMS = 1000000  # e.g. 100 for quick smoke test
    GLOBAL_SEED = 12345
    # ------------------------------------------------------------------- #

    for cfg_settings in RUN_CONFIGS:
        print(f"\n=== Running {cfg_settings['config_module']} -> {cfg_settings['output_dir']} ===")
        cfg_module = importlib.import_module(cfg_settings['config_module'])
        run_generation(
            cfg = cfg_module,
            vqvae_ckpt = Path(VQVAE_CKPT),
            ldm_ckpt = Path(cfg_settings['ldm_ckpt']),
            output_dir = Path(cfg_settings['output_dir']),
            batch_size = BATCH_SIZE,
            samples_per_condition = SAMPLES_PER_CONDITION,
            guidance_scale = GUIDANCE_SCALE,
            num_inference_steps = NUM_INFERENCE_STEPS,
            sampler_method = SAMPLER_METHOD,
            sampler_eta = SAMPLER_ETA,
            skip_existing = SKIP_EXISTING,
            limit_num_items = LIMIT_NUM_ITEMS,
            seed = GLOBAL_SEED,
        )
