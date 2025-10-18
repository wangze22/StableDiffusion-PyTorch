import csv
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
import config.celebhq_params as cfg
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.diffusion_utils import drop_image_condition, drop_text_condition
from utils.text_utils import get_text_representation, get_tokenizer_and_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Provide backwards-compatible defaults for optional config entries
if not hasattr(cfg, 'text_condition_model_name_or_path'):
    cfg.text_condition_model_name_or_path = 'openai/clip-vit-base-patch16'
if not hasattr(cfg, 'text_condition_allow_fallback'):
    cfg.text_condition_allow_fallback = False
if not hasattr(cfg, 'train_ldm_output_root'):
    cfg.train_ldm_output_root = Path('runs')
if not hasattr(cfg, 'train_ldm_save_every_epochs'):
    cfg.train_ldm_save_every_epochs = 1


@dataclass
class RunArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    logger: logging.Logger


def _latest_name(original: str) -> str:
    original_path = Path(original)
    if original_path.suffix:
        return f'{original_path.stem}_latest{original_path.suffix}'
    return f'{original_path.name}_latest'


def _plot_loss_history(loss_history: List[Dict[str, float]], logs_dir: Path) -> None:
    if not loss_history:
        return
    epochs = [item['epoch'] for item in loss_history]
    values = [item['loss'] for item in loss_history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, label='Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Training Loss')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.savefig(logs_dir / 'loss_curve.png')
    plt.close()


def _persist_loss_history(loss_history: List[Dict[str, float]], logs_dir: Path) -> None:
    if not loss_history:
        return
    csv_path = logs_dir / 'losses.csv'
    fieldnames = list(loss_history[0].keys())
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)
    _plot_loss_history(loss_history, logs_dir)


def _plot_epoch_losses(epoch_idx: int, losses: Iterable[float], logs_dir: Path) -> None:
    losses = list(losses)
    if not losses:
        return
    loss_dir = logs_dir / 'epoch_loss_plots'
    loss_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = np.arange(1, len(losses) + 1)
    plt.plot(steps, losses, label='Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Epoch {epoch_idx + 1} Losses')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.savefig(loss_dir / f'epoch_{epoch_idx + 1:03d}_losses.png')
    plt.close()


def _create_run_artifacts(task_dir: Path, output_root: Optional[Path] = None) -> RunArtifacts:
    root = Path(output_root) if output_root is not None else Path(cfg.train_ldm_output_root)
    output_root_path = root.resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_base = output_root_path / f'ddpm_{timestamp}'
    run_dir = run_base / task_dir.name
    checkpoints_dir = run_dir / 'checkpoints'
    logs_dir = run_dir / 'logs'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f'ddpm_train_{run_base.name}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(logs_dir / 'train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return RunArtifacts(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        logger=logger,
    )


def _save_model_weights(
    models: Dict[str, torch.nn.Module],
    filename_map: Dict[str, str],
    base_dir: Path,
    checkpoints_dir: Path,
    epoch_idx: int,
    save_checkpoint: bool,
) -> Dict[str, Dict[str, Path]]:
    results: Dict[str, Dict[str, Path]] = {}
    for key, model in models.items():
        target_name = filename_map[key]
        latest_path = base_dir / _latest_name(target_name)
        torch.save(model.state_dict(), latest_path)

        entry = {'latest': latest_path}
        if save_checkpoint:
            epoch_tag = f'epoch_{epoch_idx + 1:03d}'
            checkpoint_path = checkpoints_dir / f'{epoch_tag}_{target_name}'
            torch.save(model.state_dict(), checkpoint_path)
            entry['checkpoint'] = checkpoint_path
        results[key] = entry
    return results


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_text_conditioning(condition_config, logger=None):
    text_enabled = 'text' in condition_config['condition_types']
    if not text_enabled:
        if logger is not None:
            logger.info('Text conditioning disabled; skipping text encoder initialization')
        return None, None, None, 0.0, False
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
    return tokenizer, text_model, empty_embed, drop_prob, True


def train(
    data_root: Path,
    task_dir: Path,
    *,
    output_root: Optional[Path] = None,
    save_every_epochs: Optional[int] = None,
    pretrained_ldm_checkpoint: Optional[Path] = None,
    pretrained_text_encoder_checkpoint: Optional[Path] = None,
    train_imgs: Optional[int] = None,
) -> None:
    dataset_root = Path(data_root)
    task_dir = Path(task_dir).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    run_artifacts = _create_run_artifacts(task_dir, output_root=output_root)
    logger = run_artifacts.logger

    logger.info('Starting DDPM training run')
    logger.info('Dataset root: %s', dataset_root)
    logger.info('Workspace directory: %s', task_dir)
    logger.info('Artifacts directory: %s', run_artifacts.run_dir)

    condition_config = {
        'condition_types': tuple(cfg.condition_types),
        'text_condition_config': {
            'text_embed_model': cfg.text_condition_text_embed_model,
            'train_text_embed_model': cfg.text_condition_train_text_embed_model,
            'text_embed_dim': cfg.text_condition_text_embed_dim,
            'cond_drop_prob': cfg.text_condition_cond_drop_prob,
            'model_name_or_path': getattr(cfg, 'text_condition_model_name_or_path', 'openai/clip-vit-base-patch16'),
            'allow_fallback': getattr(cfg, 'text_condition_allow_fallback', False),
        },
        'image_condition_config': {
            'image_condition_input_channels': cfg.image_condition_input_channels,
            'image_condition_output_channels': cfg.image_condition_output_channels,
            'image_condition_h': cfg.image_condition_h,
            'image_condition_w': cfg.image_condition_w,
            'cond_drop_prob': cfg.image_condition_cond_drop_prob,
        },
    }
    requested_condition_types = tuple(condition_config['condition_types'])

    latent_dir = task_dir / cfg.train_vqvae_latent_dir_name
    latent_dir.mkdir(parents=True, exist_ok=True)
    vqvae_ckpt_path = task_dir / cfg.train_vqvae_autoencoder_ckpt_name

    set_seed(cfg.train_seed)
    scheduler = LinearNoiseScheduler(
        num_timesteps=cfg.diffusion_num_timesteps,
        beta_start=cfg.diffusion_beta_start,
        beta_end=cfg.diffusion_beta_end,
    )

    text_tokenizer, text_model, empty_text_embed, text_drop_prob, text_enabled = prepare_text_conditioning(
        condition_config,
        logger,
    )
    condition_types = tuple(
        ct for ct in requested_condition_types if not (ct == 'text' and not text_enabled)
    )
    condition_config['condition_types'] = condition_types
    raw_train_text_encoder = bool(condition_config['text_condition_config'].get('train_text_embed_model', False))
    train_text_encoder = raw_train_text_encoder and 'text' in condition_types and text_model is not None
    mask_drop_prob = float(condition_config['image_condition_config'].get('cond_drop_prob', 0.0)) \
        if 'image' in condition_types else 0.0
    logger.info(
        'Conditioning setup | text=%s (train=%s, drop_prob=%.3f) image=%s (drop_prob=%.3f)',
        'text' in condition_types,
        train_text_encoder,
        text_drop_prob,
        'image' in condition_types,
        mask_drop_prob,
    )

    celebhq_dataset = CelebDataset(
        split='train',
        im_path=str(dataset_root),
        im_size=cfg.dataset_im_size,
        im_channels=cfg.dataset_im_channels,
        use_latents=True,  # Default flow relies on precomputed latents
        latent_path=str(latent_dir),
        condition_config=condition_config,
    )
    dataset_size = len(celebhq_dataset)
    logger.info('Dataset size: %d', dataset_size)

    dataset_for_loader = celebhq_dataset
    if train_imgs is not None and train_imgs > 0:
        limit = min(train_imgs, dataset_size)
        if limit < dataset_size:
            indices = torch.randperm(dataset_size)[:limit].tolist()
            dataset_for_loader = Subset(celebhq_dataset, indices)
            logger.info('Limiting dataset to %d images for training/debug.', limit)
        else:
            logger.info('Requested subset (%d) >= dataset size; using full dataset.', train_imgs)

    data_loader = DataLoader(
        dataset_for_loader,
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
        logger.info('Loaded VQ-VAE checkpoint from %s', vqvae_ckpt_path)

    optimizer = Adam(model.parameters(), lr=cfg.train_ldm_lr)
    criterion = nn.MSELoss()

    num_epochs = cfg.train_ldm_epochs
    default_save_every = getattr(cfg, 'train_ldm_save_every_epochs', 1)
    save_every_epochs = max(1, int(save_every_epochs or default_save_every))
    logger.info('Configured to save checkpoints every %d epochs', save_every_epochs)

    loss_history: List[Dict[str, float]] = []

    if pretrained_ldm_checkpoint:
        checkpoint_path = Path(pretrained_ldm_checkpoint)
        if checkpoint_path.exists():
            logger.info('Loading pre-trained LDM weights from %s', checkpoint_path)
            state_dict = torch.load(checkpoint_path, map_location=device)
            load_result = model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning('Missing keys when loading LDM weights: %s', load_result.missing_keys)
            if load_result.unexpected_keys:
                logger.warning('Unexpected keys when loading LDM weights: %s', load_result.unexpected_keys)
        else:
            logger.warning('Pre-trained LDM checkpoint not found at %s', checkpoint_path)

    if text_model is not None and pretrained_text_encoder_checkpoint:
        text_ckpt_path = Path(pretrained_text_encoder_checkpoint)
        if text_ckpt_path.exists():
            logger.info('Loading pre-trained text encoder weights from %s', text_ckpt_path)
            state_dict = torch.load(text_ckpt_path, map_location=device)
            load_result = text_model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning('Missing keys when loading text encoder weights: %s', load_result.missing_keys)
            if load_result.unexpected_keys:
                logger.warning('Unexpected keys when loading text encoder weights: %s', load_result.unexpected_keys)
        else:
            logger.warning('Pre-trained text encoder checkpoint not found at %s', text_ckpt_path)

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
        logger.info('Epoch %d/%d | Loss: %.4f', epoch_idx + 1, num_epochs, mean_loss)

        loss_history.append({'epoch': epoch_idx + 1, 'loss': mean_loss})
        _persist_loss_history(loss_history, run_artifacts.logs_dir)
        _plot_epoch_losses(epoch_idx, epoch_losses, run_artifacts.logs_dir)

        models_to_save: Dict[str, torch.nn.Module] = {'ldm': model}
        filename_map: Dict[str, str] = {'ldm': cfg.train_ldm_ckpt_name}
        if train_text_encoder and text_model is not None:
            models_to_save['text_encoder'] = text_model
            filename_map['text_encoder'] = cfg.train_text_encoder_ckpt_name

        save_checkpoint = ((epoch_idx + 1) % save_every_epochs == 0) or (epoch_idx + 1 == num_epochs)
        save_results = _save_model_weights(
            models=models_to_save,
            filename_map=filename_map,
            base_dir=run_artifacts.run_dir,
            checkpoints_dir=run_artifacts.checkpoints_dir,
            epoch_idx=epoch_idx,
            save_checkpoint=save_checkpoint,
        )
        saved_messages = []
        for key, paths in save_results.items():
            latest_path = paths['latest']
            checkpoint_path = paths.get('checkpoint')
            if checkpoint_path:
                saved_messages.append(f'{key}: latest={latest_path} checkpoint={checkpoint_path}')
            else:
                saved_messages.append(f'{key}: latest={latest_path}')
        logger.info('Saved weights: %s', '; '.join(saved_messages))

    logger.info('Training complete. Artifacts stored in %s', run_artifacts.run_dir)


if __name__ == '__main__':
    output_root = getattr(cfg, 'train_ldm_output_root', Path('runs'))
    save_every_epochs = getattr(cfg, 'train_ldm_save_every_epochs', 1)
    pretrained_ldm_checkpoint = None  # e.g., Path('runs/ddpm_20251018-222220/celebhq/ddpm_ckpt_text_image_cond_clip_latest.pth')
    pretrained_text_encoder_checkpoint = None  # e.g., Path('runs/ddpm_20251018-222220/celebhq/text_encoder_ckpt_latest.pth')
    train_imgs = 128  # e.g., 128 to debug with a small subset

    train(
        data_root=cfg.dataset_im_path,
        task_dir=Path(cfg.train_task_name),
        output_root=output_root,
        save_every_epochs=save_every_epochs,
        pretrained_ldm_checkpoint=pretrained_ldm_checkpoint,
        pretrained_text_encoder_checkpoint=pretrained_text_encoder_checkpoint,
        train_imgs=train_imgs,
    )
