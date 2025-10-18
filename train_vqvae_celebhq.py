import csv
import logging
import os
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class RunArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    samples_dir: Path
    logs_dir: Path
    logger: logging.Logger


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directories(root: Path) -> RunArtifacts:
    run_dir = root
    checkpoints_dir = run_dir / 'checkpoints'
    samples_dir = run_dir / 'samples'
    logs_dir = run_dir / 'logs'

    for path in (run_dir, checkpoints_dir, samples_dir, logs_dir):
        path.mkdir(parents = True, exist_ok = True)

    logger = logging.getLogger('vqvae_train')
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
        run_dir = run_dir,
        checkpoints_dir = checkpoints_dir,
        samples_dir = samples_dir,
        logs_dir = logs_dir,
        logger = logger,
        )


def persist_loss_history(loss_history: List[Dict[str, float]], logs_dir: Path) -> None:
    if not loss_history:
        return

    csv_path = logs_dir / 'losses.csv'
    fieldnames = list(loss_history[0].keys())
    with csv_path.open('w', newline = '') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)

    epochs = [item['epoch'] for item in loss_history]
    plt.figure(figsize = (10, 6))
    for key in fieldnames:
        if key == 'epoch' or key == 'total_loss':
            continue
        plt.plot(epochs, [item[key] for item in loss_history], label = key)
    plt.plot(epochs, [item['total_loss'] for item in loss_history], label = 'total_loss', linestyle = '--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VQ-VAE Training Losses')
    plt.legend()
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(logs_dir / 'loss_curve.png')
    plt.close()


def save_epoch_comparisons(
        epoch_idx: int,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        samples_dir: Path,
        ) -> None:
    if not samples:
        return

    epoch_dir = samples_dir / f'epoch_{epoch_idx + 1:03d}'
    epoch_dir.mkdir(parents = True, exist_ok = True)

    limited_samples = samples[:10]
    if not limited_samples:
        return

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.clamp(tensor, -1.0, 1.0)
        return (tensor + 1.0) / 2.0

    inputs = torch.stack([_prepare(inp) for inp, _ in limited_samples], dim = 0)
    outputs = torch.stack([_prepare(out) for _, out in limited_samples], dim = 0)
    combined = torch.cat([inputs, outputs], dim = 0)
    grid = make_grid(combined, nrow = len(limited_samples))
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(epoch_dir / f'epoch_{epoch_idx + 1:03d}_comparisons.png')
    img.close()


def plot_epoch_losses(
        epoch_idx: int,
        step_losses: Dict[str, List[float]],
        logs_dir: Path,
        ) -> None:
    if not step_losses:
        return

    loss_dir = logs_dir / 'epoch_loss_plots'
    loss_dir.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (12, 8))
    has_data = False
    display_names = {
        'recon_loss'        : 'Reconstruction',
        'perceptual_loss'   : 'Perceptual',
        'codebook_loss'     : 'Codebook',
        'commitment_loss'   : 'Commitment',
        'generator_adv_loss': 'Generator Adversarial',
        'discriminator_loss': 'Discriminator',
        'total_loss'        : 'Total',
        }

    for key, values in step_losses.items():
        if not values:
            continue
        steps = np.arange(1, len(values) + 1)
        plt.plot(steps, values, label = display_names.get(key, key))
        has_data = True

    if not has_data:
        plt.close()
        return

    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Epoch {epoch_idx + 1} Losses')
    plt.legend()
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(loss_dir / f'epoch_{epoch_idx + 1:03d}_losses.png')
    plt.close()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents = True, exist_ok = True)


def save_weights(
        vqvae: VQVAE,
        discriminator: Discriminator,
        train_config: Dict[str, str],
        run_artifacts: RunArtifacts,
        epoch_idx: int,
        ) -> Dict[str, Path]:
    base_dir = run_artifacts.run_dir
    ensure_directory(base_dir)

    def _latest_name(original: str) -> str:
        original_path = Path(original)
        if original_path.suffix:
            return f'{original_path.stem}_latest{original_path.suffix}'
        return f'{original_path.name}_latest'

    vqvae_base_path = base_dir / _latest_name(train_config['vqvae_autoencoder_ckpt_name'])
    discriminator_base_path = base_dir / _latest_name(train_config['vqvae_discriminator_ckpt_name'])

    torch.save(vqvae.state_dict(), vqvae_base_path)
    torch.save(discriminator.state_dict(), discriminator_base_path)

    checkpoints_dir = run_artifacts.checkpoints_dir
    ensure_directory(checkpoints_dir)
    epoch_tag = f'epoch_{epoch_idx + 1:03d}'

    vqvae_epoch_path = checkpoints_dir / f'{epoch_tag}_{train_config["vqvae_autoencoder_ckpt_name"]}'
    discriminator_epoch_path = checkpoints_dir / f'{epoch_tag}_{train_config["vqvae_discriminator_ckpt_name"]}'

    torch.save(vqvae.state_dict(), vqvae_epoch_path)
    torch.save(discriminator.state_dict(), discriminator_epoch_path)

    return {
        'vqvae'               : vqvae_epoch_path,
        'discriminator'       : discriminator_epoch_path,
        'vqvae_latest'        : vqvae_base_path,
        'discriminator_latest': discriminator_base_path,
        }


def load_weights(
        vqvae_checkpoint_path: Path,
        discriminator_checkpoint_path: Path,
        vqvae: VQVAE,
        discriminator: Discriminator,
        ) -> None:
    vqvae.load_state_dict(torch.load(vqvae_checkpoint_path, map_location = device))
    discriminator.load_state_dict(torch.load(discriminator_checkpoint_path, map_location = device))


def infer_epoch_from_path(path: Path) -> Optional[int]:
    match = re.search(r'epoch_(\d+)', str(path))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def train(
        config_path: str,
        output_root: str,
        save_every_epochs: int,
        resume_vqvae_checkpoint: Optional[str] = None,
        resume_discriminator_checkpoint: Optional[str] = None,
        start_epoch: Optional[int] = None,
        train_imgs: Optional[int] = None,
        ) -> None:
    config_path = Path(config_path)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents = True, exist_ok = True)

    run_artifacts = create_run_directories(output_root_path)
    logger = run_artifacts.logger

    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise RuntimeError(f'Unable to parse config file {config_path}') from exc

    dataset_config = deepcopy(config['dataset_params'])
    autoencoder_config = deepcopy(config['autoencoder_params'])
    train_config = deepcopy(config['train_params'])

    if dataset_config.get('name') != 'celebhq':
        raise ValueError('This script only supports the CelebHQ dataset configuration.')

    setup_seed(train_config['seed'])

    train_config['resumed_from'] = {
        'vqvae'        : resume_vqvae_checkpoint,
        'discriminator': resume_discriminator_checkpoint,
        }

    with (run_artifacts.logs_dir / 'config_snapshot.yaml').open('w') as snapshot_file:
        yaml.safe_dump(
            {
                'config_path'       : str(config_path),
                'dataset_params'    : dataset_config,
                'autoencoder_params': autoencoder_config,
                'train_params'      : train_config,
                },
            snapshot_file,
            )

    logger.info('Starting VQ-VAE training')
    logger.info('Run directory: %s', run_artifacts.run_dir)

    vqvae = VQVAE(im_channels = dataset_config['im_channels'], model_config = autoencoder_config).to(device)
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels = dataset_config['im_channels']).to(device)

    im_dataset = CelebDataset(
        split = 'train',
        im_path = dataset_config['im_path'],
        im_size = dataset_config['im_size'],
        im_channels = dataset_config['im_channels'],
        )

    if train_imgs is not None and train_imgs > 0:
        limit = min(train_imgs, len(im_dataset))
        indices = torch.randperm(len(im_dataset))[:limit].tolist()
        im_dataset = torch.utils.data.Subset(im_dataset, indices)
        logger.info('Limiting dataset to %d images for training/debug.', limit)

    data_loader = DataLoader(
        im_dataset,
        batch_size = train_config['autoencoder_batch_size'],
        shuffle = True,
        num_workers = 0,
        drop_last = False,
        )

    recon_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.MSELoss()

    optimizer_d = Adam(discriminator.parameters(), lr = train_config['autoencoder_lr'], betas = (0.5, 0.999))
    optimizer_g = Adam(vqvae.parameters(), lr = train_config['autoencoder_lr'], betas = (0.5, 0.999))

    min_lr = train_config.get('autoencoder_min_lr', train_config['autoencoder_lr'] * 0.1)
    scheduler_kwargs = {
        'T_max': max(1, train_config['autoencoder_epochs']),
        'eta_min': min_lr,
    }
    scheduler_g = CosineAnnealingLR(optimizer_g, **scheduler_kwargs)
    scheduler_d = CosineAnnealingLR(optimizer_d, **scheduler_kwargs)

    disc_step_start = train_config['disc_start']
    step_count = 0
    start_epoch = 0 if start_epoch is None else max(0, start_epoch)

    num_batches_per_epoch = len(data_loader)

    if resume_vqvae_checkpoint is not None or resume_discriminator_checkpoint is not None:
        if resume_vqvae_checkpoint is None or resume_discriminator_checkpoint is None:
            raise ValueError('Both VQVAE and discriminator checkpoints must be provided to resume training.')
        resume_vqvae_path = Path(resume_vqvae_checkpoint)
        resume_discriminator_path = Path(resume_discriminator_checkpoint)
        if not resume_vqvae_path.exists():
            raise FileNotFoundError(f'VQVAE checkpoint not found at {resume_vqvae_path}')
        if not resume_discriminator_path.exists():
            raise FileNotFoundError(f'Discriminator checkpoint not found at {resume_discriminator_path}')
        logger.info('Resuming VQ-VAE from checkpoint: %s', resume_vqvae_path)
        logger.info('Resuming discriminator from checkpoint: %s', resume_discriminator_path)
        load_weights(resume_vqvae_path, resume_discriminator_path, vqvae, discriminator)
        inferred_epoch = infer_epoch_from_path(resume_vqvae_path)
        if inferred_epoch is None:
            inferred_epoch = infer_epoch_from_path(resume_discriminator_path)
        if inferred_epoch is not None and start_epoch == 0:
            start_epoch = inferred_epoch
        step_count = start_epoch * num_batches_per_epoch
        disc_step_start = 0  # ensure GAN losses are used when resuming
        logger.info('Resumed training from epoch index %d (next epoch %d)', start_epoch, start_epoch + 1)
        if start_epoch > 0:
            scheduler_g.step(start_epoch)
            scheduler_d.step(start_epoch)

    num_epochs = train_config['autoencoder_epochs']

    loss_history: List[Dict[str, float]] = []

    for epoch_idx in range(start_epoch, num_epochs):
        vqvae.train()
        discriminator.train()

        epoch_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        recon_losses: List[float] = []
        perceptual_losses: List[float] = []
        codebook_losses: List[float] = []
        commitment_losses: List[float] = []
        generator_adv_losses: List[float] = []
        discriminator_losses: List[float] = []
        total_losses: List[float] = []

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for batch in tqdm(data_loader, desc = f'Epoch {epoch_idx + 1}/{num_epochs}', leave = False):
            if isinstance(batch, (list, tuple)):
                im = batch[0]
            else:
                im = batch

            step_count += 1
            im = im.float().to(device)

            output, _, quantize_losses = vqvae(im)

            if len(epoch_samples) < 10:
                remaining = 10 - len(epoch_samples)
                sample_count = min(remaining, output.shape[0])
                if sample_count > 0:
                    inputs_cpu = im[:sample_count].detach().cpu()
                    outputs_cpu = output[:sample_count].detach().cpu()
                    epoch_samples.extend(zip(inputs_cpu, outputs_cpu))

            recon_loss = recon_criterion(output, im)
            codebook_loss = train_config['codebook_weight'] * quantize_losses['codebook_loss']
            commitment_loss = train_config['commitment_beta'] * quantize_losses['commitment_loss']
            lpips_loss = torch.mean(lpips_model(output, im))
            perceptual_loss = train_config['perceptual_weight'] * lpips_loss

            adv_loss = torch.tensor(0.0, device = device)
            use_gan = step_count > disc_step_start
            if use_gan:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = disc_criterion(
                    disc_fake_pred,
                    torch.ones_like(disc_fake_pred, device = disc_fake_pred.device),
                    )
                adv_loss = train_config['disc_weight'] * disc_fake_loss

            total_generator_loss = recon_loss + codebook_loss + commitment_loss + perceptual_loss + adv_loss
            total_generator_loss.backward()

            recon_losses.append(recon_loss.item())
            codebook_losses.append(codebook_loss.item())
            commitment_losses.append(commitment_loss.item())
            perceptual_losses.append(perceptual_loss.item())
            generator_adv_losses.append(adv_loss.item())
            total_losses.append(total_generator_loss.item())

            if use_gan:
                fake_images = output.detach()
                disc_fake_pred = discriminator(fake_images)
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(
                    disc_fake_pred,
                    torch.zeros_like(disc_fake_pred, device = disc_fake_pred.device),
                    )
                disc_real_loss = disc_criterion(
                    disc_real_pred,
                    torch.ones_like(disc_real_pred, device = disc_real_pred.device),
                    )
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) * 0.5
                disc_loss.backward()
                discriminator_losses.append(disc_loss.item())

            optimizer_g.step()
            optimizer_g.zero_grad()
            if use_gan:
                optimizer_d.step()
                optimizer_d.zero_grad()

        epoch_recon_loss = float(np.mean(recon_losses)) if recon_losses else 0.0
        epoch_perc_loss = float(np.mean(perceptual_losses)) if perceptual_losses else 0.0
        epoch_codebook_loss = float(np.mean(codebook_losses)) if codebook_losses else 0.0
        epoch_commitment_loss = float(np.mean(commitment_losses)) if commitment_losses else 0.0
        epoch_gen_adv_loss = float(np.mean(generator_adv_losses)) if generator_adv_losses else 0.0
        epoch_disc_loss = float(np.mean(discriminator_losses)) if discriminator_losses else 0.0
        epoch_total_loss = float(np.mean(total_losses)) if total_losses else 0.0

        loss_entry = {
            'epoch'             : epoch_idx + 1,
            'recon_loss'        : epoch_recon_loss,
            'perceptual_loss'   : epoch_perc_loss,
            'codebook_loss'     : epoch_codebook_loss,
            'commitment_loss'   : epoch_commitment_loss,
            'generator_adv_loss': epoch_gen_adv_loss,
            'discriminator_loss': epoch_disc_loss,
            'total_loss'        : epoch_total_loss,
            }
        loss_history.append(loss_entry)
        persist_loss_history(loss_history, run_artifacts.logs_dir)
        plot_epoch_losses(
            epoch_idx = epoch_idx,
            step_losses = {
                'recon_loss'        : recon_losses,
                'perceptual_loss'   : perceptual_losses,
                'codebook_loss'     : codebook_losses,
                'commitment_loss'   : commitment_losses,
                'generator_adv_loss': generator_adv_losses,
                'discriminator_loss': discriminator_losses,
                'total_loss'        : total_losses,
                },
            logs_dir = run_artifacts.logs_dir,
            )
        save_epoch_comparisons(epoch_idx, epoch_samples, run_artifacts.samples_dir)

        current_lr = optimizer_g.param_groups[0]['lr']
        logger.info(
            'Epoch %d/%d | Recon: %.4f | Perc: %.4f | Codebook: %.4f | Commit: %.4f | G_adv: %.4f | D: %.4f | LR: %.6f',
            epoch_idx + 1,
            num_epochs,
            epoch_recon_loss,
            epoch_perc_loss,
            epoch_codebook_loss,
            epoch_commitment_loss,
            epoch_gen_adv_loss,
            epoch_disc_loss,
            current_lr,
            )

        scheduler_g.step()
        scheduler_d.step()

        should_save = ((epoch_idx + 1) % save_every_epochs == 0) or (epoch_idx + 1 == num_epochs)
        if should_save:
            checkpoint_paths = save_weights(
                vqvae = vqvae,
                discriminator = discriminator,
                train_config = train_config,
                run_artifacts = run_artifacts,
                epoch_idx = epoch_idx,
                )
            logger.info(
                'Saved checkpoints: latest_vqvae=%s latest_disc=%s epoch_vqvae=%s epoch_disc=%s',
                checkpoint_paths['vqvae_latest'],
                checkpoint_paths['discriminator_latest'],
                checkpoint_paths['vqvae'],
                checkpoint_paths['discriminator'],
                )

    logger.info('Training complete. Artifacts stored in %s', run_artifacts.run_dir)


if __name__ == '__main__':
    config_path = 'config/celebhq.yaml'
    output_root = 'runs'
    save_every_epochs = 5
    resume_vqvae_checkpoint = fr'runs/vqvae_20251018-222220/celebhq/vqvae_autoencoder_ckpt_latest.pth'
    resume_discriminator_checkpoint = fr'runs/vqvae_20251018-222220/celebhq/vqvae_discriminator_ckpt_latest.pth'
    train_imgs = None  # e.g. 500 to debug with a subset

    train(
        config_path = config_path,
        output_root = output_root,
        save_every_epochs = save_every_epochs,
        resume_vqvae_checkpoint = resume_vqvae_checkpoint,
        resume_discriminator_checkpoint = resume_discriminator_checkpoint,
        train_imgs = train_imgs,
        )
