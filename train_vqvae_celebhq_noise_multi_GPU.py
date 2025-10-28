import csv
import logging
import os
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Subset

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
# yaml no longer required for config loading; retained for optional snapshots
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from models.discriminator import Discriminator
from models.lpips import LPIPS
from models.vqvae_noise import VQVAE

from config import celebhq_vqvae as cfg

DEFAULT_BACKEND = 'gloo' if os.name == 'nt' else 'nccl'


@dataclass
class RunArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    samples_dir: Path
    logs_dir: Path
    logger: logging.Logger


try:
    mp.set_sharing_strategy('file_system')
except (RuntimeError, AttributeError):
    # Fallback when the sharing strategy is not supported on the platform.
    pass


def _init_distributed_if_needed(local_rank: int, backend: str) -> bool:
    """Initialise torch.distributed when a valid local_rank is provided."""
    if local_rank < 0:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError('Distributed training requested but CUDA is not available.')
    if not dist.is_available():
        raise RuntimeError('torch.distributed is not available in this build of PyTorch.')

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend = backend)
    return True


def _get_logger_for_worker(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def gradients_are_finite(parameters) -> bool:
    for param in parameters:
        if param.grad is None:
            continue
        if not torch.all(torch.isfinite(param.grad)):
            return False
    return True


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_directories(root: Path, logger_name: str = 'vqvae_train') -> RunArtifacts:
    run_dir = root
    checkpoints_dir = run_dir / 'checkpoints'
    samples_dir = run_dir / 'samples'
    logs_dir = run_dir / 'logs'

    for path in (run_dir, checkpoints_dir, samples_dir, logs_dir):
        path.mkdir(parents = True, exist_ok = True)

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
    total_values = [item['total_loss'] for item in loss_history]
    component_keys = [key for key in fieldnames if key not in ('epoch', 'total_loss')]

    if component_keys:
        plt.figure(figsize = (10, 6))
        for key in component_keys:
            plt.plot(epochs, [item[key] for item in loss_history], label = key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VQ-VAE Training Component Losses')
        plt.legend()
        plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
        plt.tight_layout()
        plt.savefig(logs_dir / 'loss_components.png')
        plt.close()

    if total_values:
        plt.figure(figsize = (10, 6))
        plt.plot(epochs, total_values, label = 'total_loss', linestyle = '--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VQ-VAE Training Total Loss')
        plt.legend()
        plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
        plt.tight_layout()
        plt.savefig(logs_dir / 'loss_total.png')
        plt.close()


def save_epoch_comparisons(
        epoch_tag,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        samples_dir: Path,
        ) -> None:
    if not samples:
        return

    epoch_dir = samples_dir / f'{epoch_tag}'
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
    img.save(epoch_dir / f'{epoch_tag}_comparisons.png')
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

    display_names = {
        'recon_loss'        : 'Reconstruction',
        'perceptual_loss'   : 'Perceptual',
        'codebook_loss'     : 'Codebook',
        'commitment_loss'   : 'Commitment',
        'generator_adv_loss': 'Generator Adversarial',
        'discriminator_loss': 'Discriminator',
        'total_loss'        : 'Total',
        }

    total_key = 'total_loss'
    component_items = [(key, values) for key, values in step_losses.items() if key != total_key and values]
    total_values = step_losses.get(total_key, [])

    if component_items:
        plt.figure(figsize = (12, 8))
        for key, values in component_items:
            steps = np.arange(1, len(values) + 1)
            plt.plot(steps, values, label = display_names.get(key, key))
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch_idx + 1} Component Losses')
        plt.legend()
        plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
        plt.tight_layout()
        plt.savefig(loss_dir / f'epoch_{epoch_idx + 1:03d}_component_losses.png')
        plt.close()

    if total_values:
        plt.figure(figsize = (12, 8))
        steps = np.arange(1, len(total_values) + 1)
        plt.plot(steps, total_values, label = display_names.get(total_key, total_key))
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch_idx + 1} Total Loss')
        plt.legend()
        plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
        plt.tight_layout()
        plt.savefig(loss_dir / f'epoch_{epoch_idx + 1:03d}_total_loss.png')
        plt.close()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents = True, exist_ok = True)


def save_weights(
        vqvae: VQVAE,
        discriminator: Discriminator,
        train_config: Dict[str, str],
        run_artifacts: RunArtifacts,
        epoch_idx: int,
        n_scale,
        save_epoch: bool = True,
        ) -> Dict[str, Optional[Path]]:
    base_dir = run_artifacts.run_dir
    ensure_directory(base_dir)

    def _latest_name(original: str) -> str:
        original_path = Path(original)
        if original_path.suffix:
            return f'{original_path.stem}_latest{original_path.suffix}'
        return f'{original_path.name}_latest'

    vqvae_base_path = base_dir / _latest_name(train_config['vqvae_autoencoder_ckpt_name'])
    discriminator_base_path = base_dir / _latest_name(train_config['vqvae_discriminator_ckpt_name'])

    vqvae_to_save = unwrap_model(vqvae)
    discriminator_to_save = unwrap_model(discriminator)

    torch.save(vqvae_to_save.state_dict(), vqvae_base_path)
    torch.save(discriminator_to_save.state_dict(), discriminator_base_path)

    vqvae_epoch_path: Optional[Path] = None
    discriminator_epoch_path: Optional[Path] = None

    if save_epoch:
        checkpoints_dir = run_artifacts.checkpoints_dir
        ensure_directory(checkpoints_dir)
        epoch_tag = f'epoch_{epoch_idx + 1:03d}'

        vqvae_epoch_path = checkpoints_dir / f'n_scale_{n_scale:.4f}_{epoch_tag}_{train_config["vqvae_autoencoder_ckpt_name"]}'
        discriminator_epoch_path = checkpoints_dir / f'n_scale_{n_scale:.4f}_{epoch_tag}_{train_config["vqvae_discriminator_ckpt_name"]}'

        torch.save(vqvae_to_save.state_dict(), vqvae_epoch_path)
        torch.save(discriminator_to_save.state_dict(), discriminator_epoch_path)

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
        device: torch.device,
        ) -> None:
    vqvae_module = unwrap_model(vqvae)
    discriminator_module = unwrap_model(discriminator)
    vqvae_module.load_state_dict(torch.load(vqvae_checkpoint_path, map_location = device))
    discriminator_module.load_state_dict(torch.load(discriminator_checkpoint_path, map_location = device))


def train(
        output_root: Optional[str] = None,
        save_every_epochs: Optional[int] = None,
        resume_vqvae_checkpoint: Optional[str] = None,
        resume_discriminator_checkpoint: Optional[str] = None,
        train_imgs: Optional[int] = None,
        num_epochs = 100,
        n_scale_range = [0.0, 0.05],
        n_steps = 3,
        num_images = 30000,
        num_workers: int = 0,
        local_rank: int = -1,
        backend: Optional[str] = None,
        patience = None,
        ):
    n_list = torch.linspace(n_scale_range[0], n_scale_range[1], n_steps)
    backend = backend or DEFAULT_BACKEND
    distributed = _init_distributed_if_needed(local_rank, backend)
    device = torch.device('cuda', local_rank) if distributed else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rank = dist.get_rank() if distributed else 0
    is_main_process = (not distributed) or rank == 0

    if output_root is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_root_path = Path(cfg.train_vqvae_output_root) / f'vqvae_{timestamp}' / cfg.train_task_name
    else:
        output_root_path = Path(output_root)
    output_root_path.mkdir(parents = True, exist_ok = True)

    if save_every_epochs is None:
        save_every_epochs = int(cfg.train_vqvae_save_every_epochs)
    if resume_vqvae_checkpoint is None:
        resume_vqvae_checkpoint = cfg.model_paths_vqvae_autoencoder_ckpt_resume
    if resume_discriminator_checkpoint is None:
        resume_discriminator_checkpoint = cfg.model_paths_vqvae_discriminator_ckpt_resume

    if is_main_process:
        base_run_artifacts = create_run_directories(output_root_path, logger_name = 'vqvae_train_overview')
        logger = base_run_artifacts.logger
        overview_logger = logger
    else:
        base_run_artifacts = None
        logger = _get_logger_for_worker(f'vqvae_train_rank_{rank}')
        overview_logger = None

    if distributed:
        dist.barrier()

    dataset_config = deepcopy(getattr(cfg, 'dataset_config'))
    autoencoder_config = deepcopy(getattr(cfg, 'autoencoder_model_config'))
    train_config = deepcopy(getattr(cfg, 'train_config'))

    if dataset_config.get('name') != 'celebhq':
        raise ValueError('This script only supports the CelebHQ dataset configuration.')

    base_seed = train_config['seed']
    setup_seed(base_seed)

    train_config['resumed_from'] = {
        'vqvae'        : resume_vqvae_checkpoint,
        'discriminator': resume_discriminator_checkpoint,
        }
    train_config['num_workers'] = num_workers

    if is_main_process and base_run_artifacts is not None:
        with (base_run_artifacts.logs_dir / 'config_snapshot.yaml').open('w') as snapshot_file:
            yaml.safe_dump(
                {
                    'dataset_params'    : dataset_config,
                    'autoencoder_params': autoencoder_config,
                    'train_params'      : train_config,
                    },
                snapshot_file,
                )
        logger.info('Starting VQ-VAE training')
        logger.info('Run directory: %s', base_run_artifacts.run_dir)

    vqvae = VQVAE(im_channels = dataset_config['im_channels'], model_config = autoencoder_config).to(device)
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels = dataset_config['im_channels']).to(device)

    im_dataset = CelebDataset(
        split = 'train',
        im_path = dataset_config['im_path'],
        im_size = dataset_config['im_size'],
        im_channels = dataset_config['im_channels'],
        )
    if num_images is not None:
        max_samples = min(num_images, len(im_dataset))
        im_dataset = Subset(im_dataset, range(max_samples))

    if train_imgs is not None and train_imgs > 0:
        limit = min(train_imgs, len(im_dataset))
        indices = torch.randperm(len(im_dataset))[:limit].tolist()
        im_dataset = torch.utils.data.Subset(im_dataset, indices)
        if is_main_process:
            logger.info('Limiting dataset to %d images for training/debug.', limit)

    if distributed:
        setup_seed(base_seed + rank)

    train_sampler = DistributedSampler(im_dataset, shuffle = True) if distributed else None

    data_loader = DataLoader(
        im_dataset,
        batch_size = train_config['autoencoder_batch_size'],
        shuffle = not distributed,
        sampler = train_sampler,
        num_workers = num_workers,
        pin_memory = True,
        persistent_workers = num_workers > 0,
        drop_last = False,
        )

    recon_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.MSELoss()

    if resume_vqvae_checkpoint is not None or resume_discriminator_checkpoint is not None:
        resume_vqvae_path = Path(resume_vqvae_checkpoint)
        resume_discriminator_path = Path(resume_discriminator_checkpoint)
        if is_main_process:
            logger.info('Resuming VQ-VAE from checkpoint: %s', resume_vqvae_path)
            logger.info('Resuming discriminator from checkpoint: %s', resume_discriminator_path)
        load_weights(resume_vqvae_path, resume_discriminator_path, vqvae, discriminator, device)

    if distributed:
        vqvae = DDP(vqvae, device_ids = [local_rank], output_device = local_rank, find_unused_parameters = False)
        discriminator = DDP(
            discriminator,
            device_ids = [local_rank],
            output_device = local_rank,
            find_unused_parameters = False,
            broadcast_buffers = False,
            )

    optimizer_d = Adam(discriminator.parameters(), lr = train_config['autoencoder_lr'], betas = (0.5, 0.999))
    optimizer_g = Adam(vqvae.parameters(), lr = train_config['autoencoder_lr'], betas = (0.5, 0.999))

    initial_lr = float(train_config['autoencoder_lr'])
    min_lr_g = initial_lr * 1e-3
    milestone1 = max(1, int(round(num_epochs * 0.5)))
    milestone2 = max(milestone1 + 1, int(round(num_epochs * 0.75)))
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode = 'min', factor = 0.5, patience = patience, min_lr = min_lr_g)
    scheduler_d = MultiStepLR(optimizer_d, milestones = [milestone1, milestone2], gamma = 0.1)

    disc_step_start = train_config['disc_start']
    step_count = 0

    for scale_idx, n_scale in enumerate(n_list):
        current_run_artifacts: Optional[RunArtifacts] = None
        loss_history: List[Dict[str, float]] = []
        if is_main_process:
            scale_root_base = base_run_artifacts.run_dir if base_run_artifacts is not None else output_root_path
            scale_root = scale_root_base / f'n_scale_{n_scale:.4f}'
            current_run_artifacts = create_run_directories(
                scale_root,
                logger_name = f'vqvae_train_n_scale_{n_scale:.4f}',
                )
            logger = current_run_artifacts.logger
            logger.info('Starting training for n_scale: %.4f', n_scale)
        for epoch_idx in range(num_epochs):
            epoch_start_time = time.time()
            use_gan = epoch_idx >= disc_step_start
            # print(f'USE_GAN = {use_gan}')
            if distributed and train_sampler is not None:
                sampler_epoch = scale_idx * num_epochs + epoch_idx
                train_sampler.set_epoch(sampler_epoch)

            vqvae.train()
            discriminator.train()

            epoch_samples: List[Tuple[torch.Tensor, torch.Tensor]] = [] if is_main_process else []
            recon_losses: List[float] = [] if is_main_process else []
            perceptual_losses: List[float] = [] if is_main_process else []
            codebook_losses: List[float] = [] if is_main_process else []
            commitment_losses: List[float] = [] if is_main_process else []
            generator_adv_losses: List[float] = [] if is_main_process else []
            discriminator_losses: List[float] = [] if is_main_process else []
            total_losses: List[float] = [] if is_main_process else []

            epoch_metrics = {
                'recon_sum' : 0.0,
                'perc_sum'  : 0.0,
                'code_sum'  : 0.0,
                'commit_sum': 0.0,
                'adv_sum'   : 0.0,
                'disc_sum'  : 0.0,
                'total_sum' : 0.0,
                'gen_count' : 0.0,
                'disc_count': 0.0,
                }

            data_iterator = tqdm(
                data_loader,
                desc = f'Epoch {epoch_idx + 1}/{num_epochs}',
                leave = False,
                ) if is_main_process else data_loader

            optimizer_g.zero_grad(set_to_none = True)
            optimizer_d.zero_grad(set_to_none = True)
            for batch in data_iterator:
                if isinstance(batch, (list, tuple)):
                    im = batch[0]
                else:
                    im = batch

                step_count += 1
                im = im.float().to(device, non_blocking = True)

                output, _, quantize_losses = vqvae(im, n_scale)

                if is_main_process and len(epoch_samples) < 10:
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
                if use_gan:
                    disc_fake_pred = discriminator(output)
                    disc_fake_loss = disc_criterion(
                        disc_fake_pred,
                        torch.ones_like(disc_fake_pred, device = disc_fake_pred.device),
                        )
                    adv_loss = train_config['disc_weight'] * disc_fake_loss

                total_generator_loss = recon_loss + codebook_loss + commitment_loss + perceptual_loss + adv_loss
                total_generator_loss.backward()

                recon_value = recon_loss.item()
                codebook_value = codebook_loss.item()
                commitment_value = commitment_loss.item()
                perceptual_value = perceptual_loss.item()
                adv_value = adv_loss.item()
                total_value = total_generator_loss.item()

                epoch_metrics['recon_sum'] += recon_value
                epoch_metrics['code_sum'] += codebook_value
                epoch_metrics['commit_sum'] += commitment_value
                epoch_metrics['perc_sum'] += perceptual_value
                epoch_metrics['adv_sum'] += adv_value
                epoch_metrics['total_sum'] += total_value
                epoch_metrics['gen_count'] += 1.0

                # Update tqdm with running average loss for current epoch
                if is_main_process:
                    avg_loss = epoch_metrics['total_sum'] / max(1.0, epoch_metrics['gen_count'])
                    try:
                        data_iterator.set_postfix({
                            'avg_loss': f"{avg_loss:.4f}"
                        })
                    except Exception:
                        pass

                if is_main_process:
                    recon_losses.append(recon_value)
                    codebook_losses.append(codebook_value)
                    commitment_losses.append(commitment_value)
                    perceptual_losses.append(perceptual_value)
                    generator_adv_losses.append(adv_value)
                    total_losses.append(total_value)

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

                if gradients_are_finite(unwrap_model(vqvae).parameters()) and gradients_are_finite(unwrap_model(discriminator).parameters()):
                    optimizer_g.step()
                    optimizer_g.zero_grad(set_to_none = True)
                    if use_gan:
                        optimizer_d.step()
                        optimizer_d.zero_grad(set_to_none = True)
                else:
                    if is_main_process:
                        logger.warning('Skipping optimizer_g.step() at step %d due to non-finite gradients', step_count)
                    optimizer_g.zero_grad(set_to_none = True)
                    optimizer_d.zero_grad(set_to_none = True)
                    continue


            metrics_tensor = torch.tensor(
                [
                    epoch_metrics['recon_sum'],
                    epoch_metrics['perc_sum'],
                    epoch_metrics['code_sum'],
                    epoch_metrics['commit_sum'],
                    epoch_metrics['adv_sum'],
                    epoch_metrics['disc_sum'],
                    epoch_metrics['total_sum'],
                    epoch_metrics['gen_count'],
                    epoch_metrics['disc_count'],
                    ],
                dtype = torch.float64,
                device = device,
                )
            if distributed:
                dist.all_reduce(metrics_tensor, op = dist.ReduceOp.SUM)
            (
                recon_sum,
                perc_sum,
                code_sum,
                commit_sum,
                adv_sum,
                disc_sum,
                total_sum,
                gen_count,
                disc_count,
                ) = metrics_tensor.tolist()

            gen_count = max(gen_count, 1.0)
            epoch_recon_loss = float(recon_sum / gen_count)
            epoch_perc_loss = float(perc_sum / gen_count)
            epoch_codebook_loss = float(code_sum / gen_count)
            epoch_commitment_loss = float(commit_sum / gen_count)
            epoch_gen_adv_loss = float(adv_sum / gen_count)
            epoch_total_loss = float(total_sum / gen_count)
            epoch_disc_loss = float(disc_sum / disc_count) if disc_count > 0 else 0.0

            current_lr = optimizer_g.param_groups[0]['lr']

            if is_main_process and current_run_artifacts is not None:
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
                persist_loss_history(loss_history, current_run_artifacts.logs_dir)
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
                    logs_dir = current_run_artifacts.logs_dir,
                    )
                epoch_tag = f'n_scale_{n_scale:.4f}_epoch_{epoch_idx + 1:03d}'
                save_epoch_comparisons(epoch_tag, epoch_samples, current_run_artifacts.samples_dir)
                logger.info(
                    'Epoch %d/%d | Tot: %.4f | Recon: %.4f | Perc: %.4f | Codebook: %.4f | Commit: %.4f | G_adv: %.4f | D: %.4f | LR: %.4e | n_scale: %.4f',
                    epoch_idx + 1,
                    num_epochs,
                    epoch_total_loss,
                    epoch_recon_loss,
                    epoch_perc_loss,
                    epoch_codebook_loss,
                    epoch_commitment_loss,
                    epoch_gen_adv_loss,
                    epoch_disc_loss,
                    current_lr,
                    n_scale,
                    )

                should_save = ((epoch_idx + 1) % save_every_epochs == 0) or (epoch_idx + 1 == num_epochs)
                checkpoint_paths = save_weights(
                    vqvae = vqvae,
                    discriminator = discriminator,
                    train_config = train_config,
                    run_artifacts = current_run_artifacts,
                    epoch_idx = epoch_idx,
                    save_epoch = should_save,
                    n_scale = n_scale,
                    )
                if should_save:
                    logger.info(
                        'Saved checkpoints: latest_vqvae=%s latest_disc=%s epoch_vqvae=%s epoch_disc=%s',
                        checkpoint_paths['vqvae_latest'],
                        checkpoint_paths['discriminator_latest'],
                        checkpoint_paths['vqvae'],
                        checkpoint_paths['discriminator'],
                        )
                else:
                    logger.info(
                        'Updated latest checkpoints: latest_vqvae=%s latest_disc=%s',
                        checkpoint_paths['vqvae_latest'],
                        checkpoint_paths['discriminator_latest'],
                        )

                epoch_duration_minutes = (time.time() - epoch_start_time) / 60.0
                logger.info(
                    'Epoch %d/%d completed in %.2f minutes at n_scale %.4f',
                    epoch_idx + 1,
                    num_epochs,
                    epoch_duration_minutes,
                    n_scale,
                    )

            scheduler_g.step(epoch_total_loss)
            if use_gan:
                scheduler_d.step()

            if distributed:
                dist.barrier()

        if is_main_process and current_run_artifacts is not None:
            logger.info(
                'Training complete. n_scale = %s, Artifacts stored in %s',
                f'{n_scale:.4f}',
                current_run_artifacts.run_dir,
                )

    if distributed:
        dist.barrier()

    if is_main_process and base_run_artifacts is not None:
        final_logger = overview_logger if overview_logger is not None else base_run_artifacts.logger
        final_logger.info('All training complete. Artifacts stored in %s', base_run_artifacts.run_dir)


def _distributed_worker(rank: int, world_size: int, train_kwargs: Dict[str, Any]) -> None:
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    worker_kwargs = dict(train_kwargs)
    worker_kwargs['local_rank'] = rank
    worker_kwargs['backend'] = worker_kwargs.get('backend', DEFAULT_BACKEND)

    try:
        train(**worker_kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    # Use defaults from cfg; override by passing args if needed
    n_scale_range = [0.05, 0.1]
    n_steps = 2
    vqvae_checkpoint = '/home/SD_pytorch/runs_VQVAE_noise_server/vqvae_20251028-011824/celebhq/n_scale_0.0500/vqvae_autoencoder_ckpt_latest.pth'
    discriminator_checkpoint = '/home/SD_pytorch/runs_VQVAE_noise_server/vqvae_20251028-011824/celebhq/n_scale_0.0500/vqvae_discriminator_ckpt_latest.pth'
    # vqvae_checkpoint = 'model_pths/vqvae_autoencoder_ckpt_latest_converged.pth'
    # discriminator_checkpoint = 'model_pths/vqvae_discriminator_ckpt_latest_converged.pth'
    num_epochs = 200
    num_images = 1000000
    patience = 20
    num_workers = 8
    backend = DEFAULT_BACKEND
    local_rank_env = int(os.environ.get('LOCAL_RANK', -1))
    print(f'num_epochs = {num_epochs}')

    common_train_kwargs: Dict[str, Any] = {
        'num_images'                     : num_images,
        'num_epochs'                     : num_epochs,
        'n_scale_range'                  : n_scale_range,
        'n_steps'                        : n_steps,
        'save_every_epochs'              : 5,
        'resume_vqvae_checkpoint'        : vqvae_checkpoint,
        'resume_discriminator_checkpoint': discriminator_checkpoint,
        'num_workers'                    : num_workers,
        'backend'                        : backend,
        'patience'                       : patience
        }

    if local_rank_env < 0 and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        mp.spawn(
            _distributed_worker,
            args = (world_size, common_train_kwargs),
            nprocs = world_size,
            join = True,
            )
    else:
        train(local_rank = local_rank_env, **common_train_kwargs)
