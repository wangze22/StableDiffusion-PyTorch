import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from typing import Any, Dict, List

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.celeb_dataset import CelebDataset
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.unet_cond_base_relu import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from config import celebhq_text_image_cond_tc05 as cfg
from utils.config_utils import validate_image_config, validate_text_config
from utils.diffusion_utils import *
from utils.text_utils import *
from utils.train_utils import (
    ensure_directory,
    persist_loss_history,
    plot_epoch_loss_curve,
    save_config_snapshot_json,
    )
from datetime import datetime

# 量化加噪训练
from cim_qn_train.progressive_qn_train import *
import cim_layers.register_dict as reg_dict
import config.andi_config as andi_cfg

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
EMA_DECAY = 0.9999
DEFAULT_BACKEND = 'gloo' if os.name == 'nt' else 'nccl'
use_amp = True

try:
    mp.set_sharing_strategy('file_system')
except (RuntimeError, AttributeError):
    # Fallback when the sharing strategy is not supported on the platform.
    pass


# _FD_PER_WORKER_ESTIMATE = 4
# _FD_RESERVE = 32


def gen_run_dir(timestamp, train_stage, noise):
    output_root = cfg.train_ldm_output_root
    run_dir = Path(output_root) / f'ddpm_{timestamp}' / train_stage / f'{noise:.4f}'
    return run_dir


def _init_distributed_if_needed(local_rank: int, backend: str) -> bool:
    """Initialise a distributed process group when local_rank is provided."""
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


def create_run_artifacts(run_dir) -> Dict[str, Any]:
    """Prepare run/checkpoint/log directories and logger."""

    checkpoints_dir = run_dir / 'checkpoints'
    logs_dir = run_dir / 'logs'

    for path in (checkpoints_dir, logs_dir):
        ensure_directory(path)

    logger_name = f'scripts_refined_ddpm_{run_dir}'
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

    return {
        'run_dir'        : run_dir,
        'checkpoints_dir': checkpoints_dir,
        'logs_dir'       : logs_dir,
        'logger'         : logger,
        }


class LDM_AnDi(ProgressiveTrain):
    def train_model(self, num_workers, num_images: Optional[int] = None, local_rank: int = -1, backend: Optional[str] = None) -> None:
        backend = backend or DEFAULT_BACKEND
        distributed = _init_distributed_if_needed(local_rank, backend)
        device = torch.device('cuda', local_rank) if distributed else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        is_main_process = (not distributed) or dist.get_rank() == 0

        run_artifacts: Optional[Dict[str, Path]] = None
        if is_main_process:
            run_dir = gen_run_dir(timestamp = timestamp, train_stage = andi_cfg.train_stage, noise = self.noise_scale)
            run_artifacts = create_run_artifacts(run_dir)
            save_config_snapshot_json(run_artifacts['logs_dir'], cfg)
            logger: logging.Logger = run_artifacts['logger']
            logger.info('Loaded config from celebhq_text_image_cond module')
            logger.info('Run artifacts directory: %s', run_artifacts['run_dir'])
        else:
            logger = logging.getLogger(f'train_ddpm_rank_{dist.get_rank() if distributed else 0}')
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())

        save_every = max(1, int(cfg.train_ldm_save_every_epochs))
        loss_history: List[Dict[str, float]] = []
        legacy_ckpt_dir = Path(cfg.train_task_name)
        if is_main_process:
            ensure_directory(legacy_ckpt_dir)

        scheduler = LinearNoiseScheduler(
            num_timesteps = cfg.diffusion_num_timesteps,
            beta_start = cfg.diffusion_beta_start,
            beta_end = cfg.diffusion_beta_end,
            )

        condition_types: List[str] = []
        text_tokenizer = None
        text_model = None
        empty_text_embed = None
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

        if im_dataset_cls is None:
            raise ValueError(f'Unknown dataset name: {cfg.dataset_name}')

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

        sampler: Optional[DistributedSampler] = None
        if distributed:
            sampler = DistributedSampler(
                im_dataset,
                num_replicas = dist.get_world_size(),
                rank = dist.get_rank(),
                shuffle = True,
                drop_last = False,
                )

        world_size = dist.get_world_size() if distributed else 1

        # prefetch_factor = max(1, DEFAULT_PREFETCH_FACTOR)
        # if 'image' in condition_types and cfg.condition_config is not None:
        # img_cfg = cfg.condition_config.get('image_condition_config', {})
        # mask_channels = int(img_cfg.get('image_condition_input_channels', 0))
        # mask_h = int(img_cfg.get('image_condition_h', 0))
        # mask_w = int(img_cfg.get('image_condition_w', 0))
        # if mask_channels > 0 and mask_h > 0 and mask_w > 0:
        # mask_bytes_per_sample = mask_channels * mask_h * mask_w  # uint8 in dataset
        # batch_bytes = mask_bytes_per_sample * cfg.train_ldm_batch_size
        # Cap prefetch so queued host tensors stay within ~512MB budget.
        # memory_budget = 512 * 1024 * 1024
        # max_prefetch = max(1, memory_budget // max(batch_bytes, 1))
        # if prefetch_factor > max_prefetch:
        #     if is_main_process:
        #         logger.warning(
        #             'Reducing dataloader prefetch_factor from %d to %d to limit host memory usage for image conditioning.',
        #             prefetch_factor,
        #             max_prefetch,
        #         )
        #     prefetch_factor = max_prefetch

        if is_main_process:
            logger.info(
                'Dataloader workers per process: %d | world_size=%d',
                num_workers,
                world_size,
                )
            if num_workers == 0:
                logger.warning(
                    'Falling back to single-process data loading to stay within file descriptor limits.',
                    )

        dataloader_kwargs = dict(
            batch_size = cfg.train_ldm_batch_size,
            shuffle = sampler is None,
            sampler = sampler,
            pin_memory = (device.type == 'cuda'),
            num_workers = num_workers,
            persistent_workers = num_workers > 0,
            )
        if num_workers > 0:
            available_methods = mp.get_all_start_methods()
            start_method = 'spawn'
            if os.name != 'nt' and 'fork' in available_methods:
                start_method = 'fork'
            dataloader_kwargs.update(
                multiprocessing_context = mp.get_context(start_method),
                )

        data_loader = DataLoader(im_dataset, **dataloader_kwargs)

        if distributed and not isinstance(self.model, DDP):
            self.model.to(device)
            self.model.train()
            self.model = DDP(
                self.model,
                device_ids = [local_rank],
                output_device = local_rank,
                broadcast_buffers = False,
                )

        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        optimizer = Adam(self.model.parameters(), lr = cfg.train_ldm_lr)
        criterion = torch.nn.MSELoss()

        scaler = GradScaler(device = device_type, enabled = True)

        if use_amp and is_main_process:
            logger.info('Using mixed precision (CUDA AMP) for training')

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            patience = 30,
            factor = 0.5,
            min_lr = 1e-7,
            )

        autocast_kwargs = {'device_type': device_type, 'enabled': use_amp}
        if device_type == 'cuda':
            autocast_kwargs['dtype'] = torch.bfloat16

        for epoch_idx in range(cfg.train_ldm_epochs):
            epoch_start_time = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch_idx)

            epoch_losses: List[float] = []
            loss_sum = 0.0
            num_batches = 0.0
            progress_bar = tqdm(
                data_loader,
                desc = f'Epoch {epoch_idx + 1}/{cfg.train_ldm_epochs}',
                leave = False,
                disable = not is_main_process,
                )

            for data in progress_bar:
                cond_input = None
                if cfg.condition_config is not None:
                    im, cond_input = data
                else:
                    im = data

                optimizer.zero_grad(set_to_none = True)
                im = im.float().to(device, non_blocking = True)

                if cond_input is not None and 'text' in condition_types:
                    with torch.no_grad():
                        assert 'text' in cond_input, 'Conditioning type includes text but no text input found.'
                        validate_text_config(cfg.condition_config)
                        text_condition = get_text_representation(
                            cond_input['text'],
                            text_tokenizer,
                            text_model,
                            device,
                            )
                        text_condition = drop_text_condition(
                            text_condition,
                            im,
                            empty_text_embed,
                            cfg.ldm_text_condition_cond_drop_prob,
                            )
                        cond_input['text'] = text_condition

                if cond_input is not None and 'image' in condition_types:
                    assert 'image' in cond_input, 'Conditioning type includes image but no image input found.'
                    validate_image_config(cfg.condition_config)
                    cond_input_image = cond_input['image'].to(
                        device = device,
                        dtype = torch.float32,
                        non_blocking = True,
                        )
                    cond_input['image'] = drop_image_condition(
                        cond_input_image,
                        im,
                        cfg.ldm_image_condition_cond_drop_prob,
                        )

                noise = torch.randn_like(im)
                t = torch.randint(0, cfg.diffusion_num_timesteps, (im.shape[0],), device = device)
                noisy_im = scheduler.add_noise(im, noise, t)
                with autocast(**autocast_kwargs):
                    noise_pred = self.model(noisy_im, t, cond_input = cond_input)
                    loss = criterion(noise_pred, noise)

                if not torch.isfinite(loss):
                    if is_main_process:
                        logger.warning('Skipping step due to non-finite loss (NaN or Inf)')
                    optimizer.zero_grad(set_to_none = True)
                    continue

                loss_value = loss.item()
                if is_main_process:
                    epoch_losses.append(loss_value)
                    progress_bar.set_postfix(loss = f'{loss_value:.4f}')

                loss_sum += loss_value
                num_batches += 1.0

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(model_module.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    if is_main_process:
                        logger.warning('Skipping optimizer.step() due to non-finite gradients')
                    optimizer.zero_grad(set_to_none = True)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()

                # with torch.no_grad():
                #     for ema_param, param in zip(ema_model.parameters(), model_module.parameters()):
                #         ema_param.data.mul_(EMA_DECAY).add_(param.data, alpha = 1 - EMA_DECAY)

            loss_stats = torch.tensor(
                [loss_sum, num_batches],
                device = device,
                dtype = torch.float64,
                )
            if distributed:
                dist.all_reduce(loss_stats, op = dist.ReduceOp.SUM)
            total_loss, total_batches = loss_stats.tolist()
            avg_loss = float(total_loss / max(total_batches, 1.0))

            lr_scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_duration = time.time() - epoch_start_time
            if is_main_process:
                # 获取CPU内存占用（以GB为单位）
                memory_info = psutil.virtual_memory()
                memory_used_gb = memory_info.used / (1024 ** 3)
                memory_percent = memory_info.percent

                logger.info(
                    'Epoch %d/%d | Loss: %.4f | LR: %.3e | Time: %.2fmin | CPU Mem: %.2fGB (%.1f%%)',
                    epoch_idx + 1,
                    cfg.train_ldm_epochs,
                    avg_loss,
                    current_lr,
                    epoch_duration / 60,
                    memory_used_gb,
                    memory_percent,
                    )
                loss_history.append({'epoch': epoch_idx + 1, 'ldm_loss': avg_loss})
                persist_loss_history(loss_history, run_artifacts['logs_dir'])
                plot_epoch_loss_curve(epoch_idx + 1, epoch_losses, run_artifacts['logs_dir'])

                should_save = ((epoch_idx + 1) % save_every == 0) or (epoch_idx + 1 == cfg.train_ldm_epochs)
                checkpoints_dir = run_artifacts['checkpoints_dir']
                state_dict = model_module.state_dict()
                # ema_state_dict = ema_model.state_dict()
                latest_ckpt_path = run_artifacts['run_dir'] / cfg.model_paths_ldm_ckpt_name
                torch.save(state_dict, latest_ckpt_path)
                # ema_latest_ckpt_path = run_artifacts['run_dir'] / f'ema_{cfg.model_paths_ldm_ckpt_name}'
                # torch.save(ema_state_dict, ema_latest_ckpt_path)

                if should_save:
                    epoch_ckpt_path = checkpoints_dir / f'epoch_{epoch_idx + 1:03d}_{cfg.model_paths_ldm_ckpt_name}'
                    # legacy_ckpt_path = legacy_ckpt_dir / cfg.model_paths_ldm_ckpt_name
                    torch.save(state_dict, epoch_ckpt_path)
                    # torch.save(state_dict, legacy_ckpt_path)

                    # ema_epoch_ckpt_path = checkpoints_dir / f'epoch_{epoch_idx + 1:03d}_ema_{cfg.model_paths_ldm_ckpt_name}'
                    # ema_legacy_ckpt_path = legacy_ckpt_dir / f'ema_{cfg.model_paths_ldm_ckpt_name}'
                    # torch.save(ema_state_dict, ema_epoch_ckpt_path)
                    # torch.save(ema_state_dict, ema_legacy_ckpt_path)

                    logger.info(
                        'Saved checkpoints: latest=%s | epoch=%s ',
                        latest_ckpt_path,
                        epoch_ckpt_path,
                        # ema_latest_ckpt_path,
                        )

        if is_main_process and run_artifacts is not None:
            logger.info('Training complete. Artifacts stored in %s', run_artifacts['run_dir'])
            print('Done Training ...')


# =================================================================== #
# =================================================================== #
# =================================================================== #
# =================================================================== #
# =================================================================== #

# Configure launch parameters here; edit as needed before running.
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

num_images = 100000
local_rank = int(os.environ.get('LOCAL_RANK', -1))
backend = DEFAULT_BACKEND
num_workers = 8
model_paths_ldm_ckpt_resume = '/home/SD_pytorch/runs_tc05_qn_train_server/ddpm_20251026-062209/LSQ_AnDi/0.0800/ddpm_ckpt_text_image_cond_clip._glfast.pth'

# Instantiate the unet model
model = Unet(
    im_channels = cfg.autoencoder_z_channels,
    model_config = cfg.diffusion_model_config,
    )

trainer = LDM_AnDi(model = model)

trainer.convert_to_layers(
    convert_layer_type_list = reg_dict.nn_layers,
    tar_layer_type = 'layers_qn_lsq',
    noise_scale = 0.08,
    input_bit = 8,
    output_bit = 8,
    weight_bit = 4,
    )

andi_cfg.train_stage = 'LSQ'

andi_cfg.train_stage = 'LSQ_AnDi'
trainer.add_enhance_branch_LoR(
    ops_factor = 0.05,
    )

trainer.add_enhance_layers(ops_factor = 0.05)
trainer.model.load_state_dict(torch.load(model_paths_ldm_ckpt_resume))


def _distributed_worker(rank: int, world_size: int, num_images: Optional[int], backend: str) -> None:
    """Configure per-process environment and launch distributed training worker."""
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    # trainer.progressive_train(
    #     qn_cycle = andi_cfg.qn_cycle,
    #     update_layer_type_list = ['layers_qn_lsq'],
    #     start_cycle = 0,
    #     weight_bit_range = andi_cfg.qn_weight_bit_range,
    #     input_bit_range = andi_cfg.qn_feature_bit_range,
    #     output_bit_range = andi_cfg.qn_feature_bit_range,
    #     noise_scale_range = andi_cfg.qn_noise_range,
    #     num_workers = num_workers,
    #     num_images = num_images,
    #     local_rank = rank, backend = backend
    #     )
    trainer.train_model(
        num_workers = num_workers,
        num_images = num_images,
        local_rank = rank, backend = backend,
        )
    # trainer.progressive_train(
    #     qn_cycle = andi_cfg.qna_cycle,
    #     update_layer_type_list = ['layers_qn_lsq'],
    #     start_cycle = 0,
    #     weight_bit_range = andi_cfg.qna_weight_bit_range,
    #     input_bit_range = andi_cfg.qna_feature_bit_range,
    #     output_bit_range = andi_cfg.qna_feature_bit_range,
    #     noise_scale_range = andi_cfg.qna_noise_range,
    #     num_workers = num_workers,
    #     num_images = num_images,
    #     local_rank = rank, backend = backend,
    #     )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    if local_rank < 0 and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        mp.spawn(
            _distributed_worker,
            args = (world_size, num_images, backend),
            nprocs = world_size,
            join = True,
            )
    else:
        pass
