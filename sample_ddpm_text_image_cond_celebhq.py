import json
import random
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import random
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel
from utils.config_utils import *
from utils.text_utils import *
from dataset.celeb_dataset import CelebDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _as_list(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _as_str_if_path(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def load_config_from_module(module_path: str) -> Dict[str, Dict[str, Any]]:
    module = import_module(module_path)

    dataset_config = {
        'name'       : getattr(module, 'dataset_name'),
        'im_path'    : _as_str_if_path(getattr(module, 'dataset_im_path')),
        'im_channels': getattr(module, 'dataset_im_channels'),
        'im_size'    : getattr(module, 'dataset_im_size'),
    }

    diffusion_config = {
        'num_timesteps': getattr(module, 'diffusion_num_timesteps'),
        'beta_start'   : getattr(module, 'diffusion_beta_start'),
        'beta_end'     : getattr(module, 'diffusion_beta_end'),
    }

    condition_types = list(getattr(module, 'condition_types', []))
    condition_config: Optional[Dict[str, Any]] = None
    if condition_types:
        condition_config = {'condition_types': condition_types}
        if 'text' in condition_types:
            condition_config['text_condition_config'] = {
                'text_embed_model'      : getattr(module, 'text_condition_text_embed_model'),
                'train_text_embed_model': getattr(module, 'text_condition_train_text_embed_model'),
                'text_embed_dim'        : getattr(module, 'text_condition_text_embed_dim'),
                'cond_drop_prob'        : getattr(module, 'text_condition_cond_drop_prob'),
            }
        if 'image' in condition_types:
            condition_config['image_condition_config'] = {
                'image_condition_input_channels' : getattr(module, 'image_condition_input_channels'),
                'image_condition_output_channels': getattr(module, 'image_condition_output_channels'),
                'image_condition_h'              : getattr(module, 'image_condition_h'),
                'image_condition_w'              : getattr(module, 'image_condition_w'),
                'cond_drop_prob'                 : getattr(module, 'image_condition_cond_drop_prob'),
            }
        if 'class' in condition_types:
            class_config: Dict[str, Any] = {}
            if hasattr(module, 'class_condition_num_classes'):
                class_config['num_classes'] = getattr(module, 'class_condition_num_classes')
            if hasattr(module, 'class_condition_cond_drop_prob'):
                class_config['cond_drop_prob'] = getattr(module, 'class_condition_cond_drop_prob')
            if class_config:
                condition_config['class_condition_config'] = class_config

    ldm_config: Dict[str, Any] = {
        'down_channels'    : _as_list(getattr(module, 'ldm_down_channels')),
        'mid_channels'     : _as_list(getattr(module, 'ldm_mid_channels')),
        'down_sample'      : _as_list(getattr(module, 'ldm_down_sample')),
        'attn_down'        : _as_list(getattr(module, 'ldm_attn_down')),
        'time_emb_dim'     : getattr(module, 'ldm_time_emb_dim'),
        'norm_channels'    : getattr(module, 'ldm_norm_channels'),
        'num_heads'        : getattr(module, 'ldm_num_heads'),
        'conv_out_channels': getattr(module, 'ldm_conv_out_channels'),
        'num_down_layers'  : getattr(module, 'ldm_num_down_layers'),
        'num_mid_layers'   : getattr(module, 'ldm_num_mid_layers'),
        'num_up_layers'    : getattr(module, 'ldm_num_up_layers'),
    }
    if condition_config is not None:
        ldm_config['condition_config'] = condition_config

    autoencoder_config = {
        'z_channels'   : getattr(module, 'autoencoder_z_channels'),
        'codebook_size': getattr(module, 'autoencoder_codebook_size'),
        'down_channels': _as_list(getattr(module, 'autoencoder_down_channels')),
        'mid_channels' : _as_list(getattr(module, 'autoencoder_mid_channels')),
        'down_sample'  : _as_list(getattr(module, 'autoencoder_down_sample')),
        'attn_down'    : _as_list(getattr(module, 'autoencoder_attn_down')),
        'norm_channels': getattr(module, 'autoencoder_norm_channels'),
        'num_heads'    : getattr(module, 'autoencoder_num_heads'),
        'num_down_layers': getattr(module, 'autoencoder_num_down_layers'),
        'num_mid_layers' : getattr(module, 'autoencoder_num_mid_layers'),
        'num_up_layers'  : getattr(module, 'autoencoder_num_up_layers'),
    }

    train_config: Dict[str, Any] = {
        'seed'                     : getattr(module, 'train_seed'),
        'task_name'                : getattr(module, 'train_task_name'),
        'ldm_batch_size'           : getattr(module, 'train_ldm_batch_size'),
        'autoencoder_batch_size'   : getattr(module, 'train_autoencoder_batch_size'),
        'disc_start'               : getattr(module, 'train_disc_start'),
        'disc_weight'              : getattr(module, 'train_disc_weight'),
        'codebook_weight'          : getattr(module, 'train_codebook_weight'),
        'commitment_beta'          : getattr(module, 'train_commitment_beta'),
        'perceptual_weight'        : getattr(module, 'train_perceptual_weight'),
        'kl_weight'                : getattr(module, 'train_kl_weight'),
        'ldm_epochs'               : getattr(module, 'train_ldm_epochs'),
        'autoencoder_epochs'       : getattr(module, 'train_autoencoder_epochs'),
        'num_samples'              : getattr(module, 'train_num_samples'),
        'num_grid_rows'            : getattr(module, 'train_num_grid_rows'),
        'ldm_lr'                   : getattr(module, 'train_ldm_lr'),
        'autoencoder_lr'           : getattr(module, 'train_autoencoder_lr'),
        'autoencoder_acc_steps'    : getattr(module, 'train_autoencoder_acc_steps'),
        'autoencoder_img_save_steps': getattr(module, 'train_autoencoder_img_save_steps'),
        'save_latents'             : getattr(module, 'train_save_latents'),
        'cf_guidance_scale'        : getattr(module, 'train_cf_guidance_scale'),
        'vae_latent_dir_name'      : getattr(module, 'train_vae_latent_dir_name'),
        'vqvae_latent_dir_name'    : getattr(module, 'train_vqvae_latent_dir_name'),
        'ldm_output_root'          : _as_str_if_path(getattr(module, 'train_ldm_output_root')),
        'ldm_save_every_epochs'    : getattr(module, 'train_ldm_save_every_epochs'),
        'ldm_ckpt_name'            : _as_str_if_path(getattr(module, 'train_ldm_ckpt_name')),
        'vqvae_autoencoder_ckpt_name': _as_str_if_path(getattr(module, 'train_vqvae_autoencoder_ckpt_name')),
    }

    if hasattr(module, 'train_text_encoder_ckpt_name'):
        text_encoder_ckpt = getattr(module, 'train_text_encoder_ckpt_name')
        if text_encoder_ckpt:
            train_config['text_encoder_ckpt_name'] = _as_str_if_path(text_encoder_ckpt)

    return {
        'dataset_params'    : dataset_config,
        'diffusion_params'  : diffusion_config,
        'ldm_params'        : ldm_config,
        'autoencoder_params': autoencoder_config,
        'train_params'      : train_config,
    }


def _load_random_mask(dataset_root: Path,
                      mask_channels: int,
                      mask_h: int,
                      mask_w: int) -> torch.Tensor:
    mask_dir = dataset_root / 'CelebAMask-HQ-mask'
    mask_files = sorted(mask_dir.glob('*.png'))
    if not mask_files:
        raise FileNotFoundError(f'No mask files found in {mask_dir}')
    mask_path = random.choice(mask_files)
    mask_im = Image.open(mask_path)
    mask_np = np.array(mask_im, dtype=np.int32)
    if mask_np.shape[0] != mask_h or mask_np.shape[1] != mask_w:
        resized = mask_im.resize((mask_w, mask_h), Image.NEAREST)
        mask_np = np.array(resized, dtype=np.int32)
    mask_im.close()

    mask_tensor = np.zeros((mask_h, mask_w, mask_channels), dtype=np.float32)
    for channel_idx in range(mask_channels):
        mask_tensor[mask_np == (channel_idx + 1), channel_idx] = 1.0
    return torch.from_numpy(mask_tensor).permute(2, 0, 1)


def _save_mask(mask_tensor: torch.Tensor, path: Path) -> None:
    """
    Persist a semantic mask (C, H, W) as a single-channel PNG.
    """
    mask_classes = torch.argmax(mask_tensor, dim=0).to(dtype=torch.uint8).cpu().numpy()
    mask_img = Image.fromarray(mask_classes, mode='L')
    mask_img.save(path)


def sample(
        model, scheduler, train_config, diffusion_model_config,
        autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model,
        samples_dir: Path
        ):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    ########### Sample random noise latent ##########
    # For not fixing generation with one sample
    xt = torch.randn(
        (
            1,
            autoencoder_model_config['z_channels'],
            im_size,
            im_size
            )
        ).to(device)
    ###############################################

    ############ Create Conditional input ###############
    text_prompt = ['She is a woman with blond hair. She is wearing lipstick.']
    empty_prompt = ['']
    text_prompt_embed = get_text_representation(
        text_prompt,
        text_tokenizer,
        text_model,
        device
        )
    # Can replace empty prompt with negative prompt
    empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
    assert empty_text_embed.shape == text_prompt_embed.shape

    condition_config = get_config_value(diffusion_model_config, key = 'condition_config', default_value = None)
    validate_image_config(condition_config)

    # This is required to get a random but valid mask
    dataset = CelebDataset(
        split = 'train',
        im_path = dataset_config['im_path'],
        im_size = dataset_config['im_size'],
        im_channels = dataset_config['im_channels'],
        use_latents = True,
        latent_path = os.path.join(
            train_config['task_name'],
            train_config['vqvae_latent_dir_name']
            ),
        condition_config = condition_config
        )
    mask_idx = random.randint(0, len(dataset.masks))
    mask = dataset.get_mask(mask_idx).unsqueeze(0).to(device)
    uncond_input = {
        'text' : empty_text_embed,
        'image': torch.zeros_like(mask)
        }
    cond_input = {
        'text' : text_prompt_embed,
        'image': mask
        }
    ###############################################

    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 0.8)

    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)

        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        if i == 0:
            # Decode ONLY the final image to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow = 10)
        img = torchvision.transforms.ToPILImage()(grid)

        img_path = samples_dir / f'x0_{i}.png'
        img.save(img_path)
        img.close()
    ##############################################################


def infer(config_module: str,
          ldm_ckpt_path: Path,
          vqvae_ckpt_path: Path,
          samples_dir: Optional[Path] = None,
          num_inference_steps: Optional[int] = None):
    config = load_config_from_module(config_module)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    ldm_ckpt_path = Path(ldm_ckpt_path)
    vqvae_ckpt_path = Path(vqvae_ckpt_path)
    if samples_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        samples_dir = ldm_ckpt_path.parent / f'cond_text_image_samples_{timestamp}'
    samples_dir = Path(samples_dir)
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ############# Validate the config #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for image and text conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no text condition found in config")
    assert 'image' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no image condition found in config")
    validate_text_config(condition_config)
    validate_image_config(condition_config)
    ###############################################
    
    ############# Load tokenizer and text model #################
    with torch.no_grad():
        # Load tokenizer and text model based on config
        # Also get empty text representation
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                             ['text_embed_model'], device=device)
    ###############################################

    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if ldm_ckpt_path.exists():
        print(f'Loaded unet checkpoint from {ldm_ckpt_path}')
        model.load_state_dict(torch.load(str(ldm_ckpt_path), map_location=device))
    else:
        raise FileNotFoundError(f'Model checkpoint {ldm_ckpt_path} not found')
    #####################################
    
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if vqvae_ckpt_path.exists():
        print(f'Loaded vae checkpoint from {vqvae_ckpt_path}')
        vae.load_state_dict(torch.load(str(vqvae_ckpt_path), map_location=device))
    else:
        raise FileNotFoundError(f'VAE checkpoint {vqvae_ckpt_path} not found')
    #####################################
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model,
               samples_dir=samples_dir)


if __name__ == '__main__':
    config_module = 'config.celebhq_params'
    vqvae_ckpt_path = Path('runs/vqvae_20251018-222220/celebhq/vqvae_autoencoder_ckpt_latest.pth')
    ldm_ckpt_path = Path('runs/ddpm_20251019-041414/celebhq/ddpm_ckpt_text_image_cond_clip_latest.pth')
    ldm_ckpt_path = Path('runs/ddpm_20251021-190501/celebhq/ddpm_ckpt_text_image_cond_clip.pth')
    samples_root = ldm_ckpt_path.parent / 'cond_text_image_samples'
    samples_dir = samples_root / datetime.now().strftime('%Y%m%d-%H%M%S')
    num_inference_steps = 1000  # Adjust this value to control sampling steps (<= config diffusion steps)

    infer(
        config_module=config_module,
        ldm_ckpt_path=ldm_ckpt_path,
        vqvae_ckpt_path=vqvae_ckpt_path,
        samples_dir=samples_dir,
        num_inference_steps=num_inference_steps,
    )
