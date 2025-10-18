import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
import config.celebhq_params as cfg
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.diffusion_utils import drop_image_condition, drop_text_condition
from utils.text_utils import get_text_representation, get_tokenizer_and_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_text_conditioning(condition_config):
    if 'text' not in condition_config['condition_types']:
        return None, None, None, 0.0
    text_cfg = condition_config.get('text_condition_config')
    if text_cfg is None:
        raise KeyError('text_condition_config missing for text conditioning')

    train_text_encoder = bool(text_cfg.get('train_text_embed_model', False))
    tokenizer, text_model = get_tokenizer_and_model(
        text_cfg['text_embed_model'],
        device=device,
        eval_mode=not train_text_encoder,
    )
    if train_text_encoder:
        text_model.train()
    drop_prob = float(text_cfg.get('cond_drop_prob', 0.0))
    with torch.no_grad():
        empty_embed = get_text_representation([''], tokenizer, text_model, device)
    return tokenizer, text_model, empty_embed, drop_prob


def train(data_root: Path, run_dir: Path) -> None:
    dataset_root = Path(data_root)
    condition_config = {
        'condition_types': tuple(cfg.condition_types),
        'text_condition_config': {
            'text_embed_model': cfg.text_condition_text_embed_model,
            'train_text_embed_model': cfg.text_condition_train_text_embed_model,
            'text_embed_dim': cfg.text_condition_text_embed_dim,
            'cond_drop_prob': cfg.text_condition_cond_drop_prob,
        },
        'image_condition_config': {
            'image_condition_input_channels': cfg.image_condition_input_channels,
            'image_condition_output_channels': cfg.image_condition_output_channels,
            'image_condition_h': cfg.image_condition_h,
            'image_condition_w': cfg.image_condition_w,
            'cond_drop_prob': cfg.image_condition_cond_drop_prob,
        },
    }
    condition_types = tuple(condition_config['condition_types'])

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = run_dir / cfg.train_vqvae_latent_dir_name
    latent_dir.mkdir(parents=True, exist_ok=True)
    ldm_ckpt_path = run_dir / cfg.train_ldm_ckpt_name
    vqvae_ckpt_path = run_dir / cfg.train_vqvae_autoencoder_ckpt_name

    set_seed(cfg.train_seed)
    scheduler = LinearNoiseScheduler(
        num_timesteps=cfg.diffusion_num_timesteps,
        beta_start=cfg.diffusion_beta_start,
        beta_end=cfg.diffusion_beta_end,
    )

    text_tokenizer, text_model, empty_text_embed, text_drop_prob = prepare_text_conditioning(condition_config)
    mask_drop_prob = float(condition_config['image_condition_config'].get('cond_drop_prob', 0.0)) \
        if 'image' in condition_types else 0.0

    celebhq_dataset = CelebDataset(
        split='train',
        im_path=str(dataset_root),
        im_size=cfg.dataset_im_size,
        im_channels=cfg.dataset_im_channels,
        use_latents=True,  # Default flow relies on precomputed latents
        latent_path=str(latent_dir),
        condition_config=condition_config,
    )

    data_loader = DataLoader(
        celebhq_dataset,
        batch_size=cfg.train_ldm_batch_size,
        shuffle=True,
    )

    model_config = {
        'down_channels': list(cfg.ldm_down_channels),
        'mid_channels': list(cfg.ldm_mid_channels),
        'down_sample': list(cfg.ldm_down_sample),
        'attn_down': list(cfg.ldm_attn_down),
        'time_emb_dim': cfg.ldm_time_emb_dim,
        'norm_channels': cfg.ldm_norm_channels,
        'num_heads': cfg.ldm_num_heads,
        'conv_out_channels': cfg.ldm_conv_out_channels,
        'num_down_layers': cfg.ldm_num_down_layers,
        'num_mid_layers': cfg.ldm_num_mid_layers,
        'num_up_layers': cfg.ldm_num_up_layers,
    }
    model_config['condition_config'] = condition_config
    model = Unet(
        im_channels=cfg.autoencoder_z_channels,
        model_config=model_config,
    ).to(device)
    model.train()

    vae = None
    if not celebhq_dataset.use_latents:
        print('Latents not found on disk. Encoding images on-the-fly with VQ-VAE.')
        vae = VQVAE(
            im_channels=cfg.dataset_im_channels,
            model_config={
                'z_channels': cfg.autoencoder_z_channels,
                'codebook_size': cfg.autoencoder_codebook_size,
                'down_channels': list(cfg.autoencoder_down_channels),
                'mid_channels': list(cfg.autoencoder_mid_channels),
                'down_sample': list(cfg.autoencoder_down_sample),
                'attn_down': list(cfg.autoencoder_attn_down),
                'norm_channels': cfg.autoencoder_norm_channels,
                'num_heads': cfg.autoencoder_num_heads,
                'num_down_layers': cfg.autoencoder_num_down_layers,
                'num_mid_layers': cfg.autoencoder_num_mid_layers,
                'num_up_layers': cfg.autoencoder_num_up_layers,
            },
        ).to(device)
        vae.eval()
        if not vqvae_ckpt_path.exists():
            raise FileNotFoundError(f'VQ-VAE checkpoint not found at {vqvae_ckpt_path}')
        vae.load_state_dict(torch.load(vqvae_ckpt_path, map_location=device))
        for param in vae.parameters():
            param.requires_grad = False

    optimizer = Adam(model.parameters(), lr=cfg.train_ldm_lr)
    criterion = nn.MSELoss()

    num_epochs = cfg.train_ldm_epochs

    for epoch_idx in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(data_loader, desc=f'Epoch {epoch_idx + 1}/{num_epochs}', leave=False):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                images, cond_inputs = batch
            else:
                images, cond_inputs = batch, {}

            optimizer.zero_grad()
            images = images.float().to(device)

            if not celebhq_dataset.use_latents and vae is not None:
                with torch.no_grad():
                    images, _ = vae.encode(images)

            batch_conditions = {}

            if 'text' in condition_types and text_tokenizer is not None:
                if 'text' not in cond_inputs:
                    raise KeyError('Expected text conditioning input but none was provided')
                with torch.no_grad():
                    text_condition = get_text_representation(
                        cond_inputs['text'],
                        text_tokenizer,
                        text_model,
                        device,
                    )
                text_condition = drop_text_condition(text_condition, images, empty_text_embed, text_drop_prob)
                batch_conditions['text'] = text_condition

            if 'image' in condition_types:
                if 'image' not in cond_inputs:
                    raise KeyError('Expected mask conditioning input but none was provided')
                mask_condition = cond_inputs['image'].to(device).float()
                mask_condition = drop_image_condition(mask_condition, images, mask_drop_prob)
                batch_conditions['image'] = mask_condition

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0,
                cfg.diffusion_num_timesteps,
                (images.shape[0],),
                device=device,
            )
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, cond_input=batch_conditions)

            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f'Epoch {epoch_idx + 1}/{num_epochs} | Loss: {mean_loss:.4f}')
        torch.save(model.state_dict(), ldm_ckpt_path)

    print('Training complete.')


if __name__ == '__main__':
    train(cfg.dataset_im_path, Path(cfg.train_task_name))
