import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

from models.unet_cond_base import Unet

from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from config import celebhq_text_image_cond as cfg
from utils.config_utils import validate_image_config, validate_text_config
from utils.diffusion_utils import *
from utils.text_utils import *
from utils.train_utils import (
    create_run_artifacts,
    ensure_directory,
    persist_loss_history,
    plot_epoch_loss_curve,
    save_config_snapshot_json,
    )

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(num_images: int = None):
    run_artifacts = create_run_artifacts({
        'task_name': cfg.train_task_name,
        'ldm_output_root': cfg.train_ldm_output_root,
        })
    # Save current config module (cfg) into a JSON snapshot (single-line helper)
    save_config_snapshot_json(run_artifacts['logs_dir'], cfg)
    logger: logging.Logger = run_artifacts['logger']
    logger.info('Loaded config from celebhq_text_image_cond module')
    logger.info('Run artifacts directory: %s', run_artifacts['run_dir'])

    save_every = max(1, int(cfg.train_ldm_save_every_epochs))
    loss_history: List[Dict[str, float]] = []
    legacy_ckpt_dir = Path(cfg.train_task_name)
    ensure_directory(legacy_ckpt_dir)

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps = cfg.diffusion_num_timesteps,
        beta_start = cfg.diffusion_beta_start,
        beta_end = cfg.diffusion_beta_end,
        )
    ###############################################

    # Instantiate Condition related components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    if cfg.condition_config is not None:
        condition_types = list(cfg.ldm_condition_types)
        if 'text' in condition_types:
            validate_text_config(cfg.condition_config)
            with torch.no_grad():
                # Load tokenizer and text model based on config
                # Also get empty text representation
                text_tokenizer, text_model = get_tokenizer_and_model(
                    cfg.ldm_text_condition_text_embed_model,
                    device = device,
                    )
                empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)

    im_dataset_cls = {
        'celebhq': CelebDataset,
        }.get(cfg.dataset_name)

    im_dataset = im_dataset_cls(
        split = 'train',
        im_path = cfg.dataset_im_path,
        im_size = cfg.dataset_im_size,
        im_channels = cfg.dataset_im_channels,
        use_latents = True,
        latent_path = cfg.train_vqvae_latent_dir_name,
        condition_config = cfg.condition_config,
        )
    if num_images is not None:
        max_samples = min(num_images, len(im_dataset))
        im_dataset = Subset(im_dataset, range(max_samples))

    data_loader = DataLoader(
        im_dataset,
        batch_size = cfg.train_ldm_batch_size,
        shuffle = True,
        )

    # Instantiate the unet model
    model = Unet(
        im_channels = cfg.autoencoder_z_channels,
        model_config = cfg.diffusion_model_config,
        ).to(device)

    # Create EMA model
    ema_model = Unet(
        im_channels = cfg.autoencoder_z_channels,
        model_config = cfg.diffusion_model_config,
        ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    resume_path = cfg.model_paths_ldm_ckpt_resume
    if resume_path is not None:
        model.load_state_dict(torch.load(str(resume_path), map_location = device))
        ema_model.load_state_dict(torch.load(str(resume_path), map_location = device))
        logger.info(f'Loaded ldm model {resume_path}')
    model.train()

    # Specify training parameters
    num_epochs = cfg.train_ldm_epochs
    optimizer = Adam(model.parameters(), lr = cfg.train_ldm_lr)
    criterion = torch.nn.MSELoss()
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled = use_amp)
    if use_amp:
        logger.info('Using mixed precision (CUDA AMP) for training')

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        patience = 10,
        factor = 0.5,
        min_lr = 1e-7,
        )

    # Run training
    for epoch_idx in range(num_epochs):
        epoch_losses: List[float] = []
        progress_bar = tqdm(data_loader, desc = f'Epoch {epoch_idx + 1}/{num_epochs}', leave = False)
        for data in progress_bar:
            cond_input = None
            if cfg.condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)

            ########### Handling Conditional Input ###########
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(cfg.condition_config)
                    text_condition = get_text_representation(
                        cond_input['text'],
                        text_tokenizer,
                        text_model,
                        device,
                        )
                    text_drop_prob = cfg.ldm_text_condition_cond_drop_prob
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(cfg.condition_config)
                cond_input_image = cond_input['image'].to(device)
                # Drop condition
                im_drop_prob = cfg.ldm_image_condition_cond_drop_prob
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)

            ################################################

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, cfg.diffusion_num_timesteps, (im.shape[0],)).to(device)
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            with autocast(device_type = 'cuda', enabled=use_amp, dtype=torch.bfloat16):
                noise_pred = model(noisy_im, t, cond_input = cond_input)
                loss = criterion(noise_pred, noise)
            epoch_losses.append(loss.item())
            progress_bar.set_postfix(loss = f'{loss.item():.4f}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update EMA model
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data = ema_param.data * 0.9999 + param.data * (1 - 0.9999)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        lr_scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            'Epoch %d/%d | Loss: %.4f | LR: %.3e',
            epoch_idx + 1,
            num_epochs,
            avg_loss,
            current_lr
            )

        loss_history.append({'epoch': epoch_idx + 1, 'ldm_loss': avg_loss})
        persist_loss_history(loss_history, run_artifacts['logs_dir'])
        plot_epoch_loss_curve(epoch_idx + 1, epoch_losses, run_artifacts['logs_dir'])

        should_save = ((epoch_idx + 1) % save_every == 0) or (epoch_idx + 1 == num_epochs)
        if should_save:
            state_dict = model.state_dict()
            ema_state_dict = ema_model.state_dict()
            checkpoints_dir = run_artifacts['checkpoints_dir']

            # Save regular model
            latest_ckpt_path = run_artifacts['run_dir'] / cfg.model_paths_ldm_ckpt_name
            epoch_ckpt_path = checkpoints_dir / f'epoch_{epoch_idx + 1:03d}_{cfg.model_paths_ldm_ckpt_name}'
            torch.save(state_dict, latest_ckpt_path)
            torch.save(state_dict, epoch_ckpt_path)
            legacy_ckpt_path = legacy_ckpt_dir / cfg.model_paths_ldm_ckpt_name
            torch.save(state_dict, legacy_ckpt_path)

            # Save EMA model
            ema_latest_ckpt_path = run_artifacts['run_dir'] / f'ema_{cfg.model_paths_ldm_ckpt_name}'
            ema_epoch_ckpt_path = checkpoints_dir / f'epoch_{epoch_idx + 1:03d}_ema_{cfg.model_paths_ldm_ckpt_name}'
            torch.save(ema_state_dict, ema_latest_ckpt_path)
            torch.save(ema_state_dict, ema_epoch_ckpt_path)
            ema_legacy_ckpt_path = legacy_ckpt_dir / f'ema_{cfg.model_paths_ldm_ckpt_name}'
            torch.save(ema_state_dict, ema_legacy_ckpt_path)

            logger.info(
                'Saved checkpoints: latest=%s | epoch=%s | ema_latest=%s',
                latest_ckpt_path,
                epoch_ckpt_path,
                ema_latest_ckpt_path,
                )

    logger.info('Training complete. Artifacts stored in %s', run_artifacts['run_dir'])
    print('Done Training ...')


if __name__ == '__main__':
    num_images = 3000000
    train(num_images = num_images)
