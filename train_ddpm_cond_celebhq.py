import csv
import logging
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class RunArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    logger: logging.Logger


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents = True, exist_ok = True)


def create_run_artifacts(output_root: Path, task_name: str) -> RunArtifacts:
    output_root = Path(output_root)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = output_root / f'ddpm_{timestamp}' / task_name
    checkpoints_dir = run_dir / 'checkpoints'
    logs_dir = run_dir / 'logs'

    for path in (run_dir, checkpoints_dir, logs_dir):
        ensure_directory(path)

    logger_name = f'ddpm_train_{timestamp}'
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
        logs_dir = logs_dir,
        logger = logger,
        )


def persist_loss_history(loss_history: List[Dict[str, float]], logs_dir: Path) -> None:
    if not loss_history:
        return

    logs_dir = Path(logs_dir)
    csv_path = logs_dir / 'losses.csv'
    fieldnames = list(loss_history[0].keys())

    with csv_path.open('w', newline = '') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)

    metrics = [key for key in fieldnames if key != 'epoch']
    if not metrics:
        return

    epochs = [item['epoch'] for item in loss_history]
    plt.figure(figsize = (10, 6))
    for key in metrics:
        plt.plot(epochs, [item[key] for item in loss_history], label = key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Training Losses')
    plt.legend()
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(logs_dir / 'loss_curve.png')
    plt.close()


def plot_epoch_loss_curve(epoch_idx: int, losses: List[float], logs_dir: Path) -> None:
    if not losses:
        return

    loss_dir = Path(logs_dir) / 'epoch_loss_plots'
    ensure_directory(loss_dir)

    steps = np.arange(1, len(losses) + 1)
    plt.figure(figsize = (10, 6))
    plt.plot(steps, losses, label = 'loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Epoch {epoch_idx + 1} Loss')
    plt.legend()
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(loss_dir / f'epoch_{epoch_idx + 1:03d}_losses.png')
    plt.close()


def _latest_checkpoint_name(base_name: str) -> str:
    base_path = Path(base_name)
    if base_path.suffix:
        return f'{base_path.stem}_latest{base_path.suffix}'
    return f'{base_path.name}_latest'


def save_model_checkpoint(
        model: torch.nn.Module,
        base_name: str,
        run_artifacts: RunArtifacts,
        epoch_idx: int,
        ) -> Dict[str, Path]:
    latest_name = _latest_checkpoint_name(base_name)
    latest_path = run_artifacts.run_dir / latest_name
    torch.save(model.state_dict(), latest_path)

    ensure_directory(run_artifacts.checkpoints_dir)
    epoch_tag = f'epoch_{epoch_idx + 1:03d}'
    epoch_path = run_artifacts.checkpoints_dir / f'{epoch_tag}_{base_name}'
    torch.save(model.state_dict(), epoch_path)

    return {
        'latest': latest_path,
        'epoch' : epoch_path,
        }


def resolve_task_relative_path(task_root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path_value = Path(value)
    if path_value.is_absolute() or len(path_value.parts) > 1:
        return path_value
    return Path(task_root) / path_value


def load_config_from_module(module_path: str) -> Dict[str, Dict[str, Any]]:
    module = import_module(module_path)

    def _as_list(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        return value

    def _as_str_if_path(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        return value

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
                'text_embed_model'       : getattr(module, 'text_condition_text_embed_model'),
                'train_text_embed_model' : getattr(module, 'text_condition_train_text_embed_model'),
                'text_embed_dim'         : getattr(module, 'text_condition_text_embed_dim'),
                'cond_drop_prob'         : getattr(module, 'text_condition_cond_drop_prob'),
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
        'ldm_ckpt_name'            : getattr(module, 'train_ldm_ckpt_name'),
        'vqvae_autoencoder_ckpt_name': _as_str_if_path(getattr(module, 'train_vqvae_autoencoder_ckpt_name')),
        }

    text_encoder_ckpt = getattr(module, 'train_text_encoder_ckpt_name', None)
    if text_encoder_ckpt:
        train_config['text_encoder_ckpt_name'] = _as_str_if_path(text_encoder_ckpt)

    return {
        'dataset_params'    : dataset_config,
        'diffusion_params'  : diffusion_config,
        'ldm_params'        : ldm_config,
        'autoencoder_params': autoencoder_config,
        'train_params'      : train_config,
        }


def infer_epoch_from_path(path: Path) -> Optional[int]:
    match = re.search(r'epoch_(\d+)', str(path))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def train(
        config_module: str,
        output_root: Optional[str] = None,
        save_every_epochs: Optional[int] = None,
        resume_checkpoint: Optional[str] = None,
        start_epoch: Optional[int] = None,
        train_imgs: Optional[int] = None,
        ) -> None:
    config = load_config_from_module(config_module)
    dataset_config = deepcopy(config['dataset_params'])
    diffusion_config = deepcopy(config['diffusion_params'])
    diffusion_model_config = deepcopy(config['ldm_params'])
    autoencoder_model_config = deepcopy(config['autoencoder_params'])
    train_config = deepcopy(config['train_params'])

    output_root_value = output_root or train_config.get('ldm_output_root', 'runs')
    output_root_path = Path(output_root_value)
    run_artifacts = create_run_artifacts(output_root_path, train_config['task_name'])
    logger = run_artifacts.logger

    logger.info('Loaded configuration module %s', config_module)
    logger.info('Run directory: %s', run_artifacts.run_dir)

    if 'seed' in train_config and train_config['seed'] is not None:
        setup_seed(train_config['seed'])
        logger.info('Seed set to %d', train_config['seed'])

    snapshot = {
        'config_module'     : config_module,
        'dataset_params'    : dataset_config,
        'diffusion_params'  : diffusion_config,
        'ldm_params'        : diffusion_model_config,
        'autoencoder_params': autoencoder_model_config,
        'train_params'      : {
            **train_config,
            'resolved_output_root': str(output_root_path),
            'run_dir'             : str(run_artifacts.run_dir),
            'save_every_override' : save_every_epochs,
            'train_imgs_override' : train_imgs,
            'resume_checkpoint'   : resume_checkpoint,
            'start_epoch_override': start_epoch,
            },
        }
    with (run_artifacts.logs_dir / 'config_snapshot.yaml').open('w') as snapshot_file:
        yaml.safe_dump(snapshot, snapshot_file)

    condition_config = get_config_value(diffusion_model_config, key = 'condition_config', default_value = None)
    condition_types = condition_config['condition_types'] if condition_config is not None else []
    logger.info('Condition types: %s', ', '.join(condition_types) if condition_types else 'none')

    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    if condition_config is not None and 'text' in condition_types:
        validate_text_config(condition_config)
        text_tokenizer, text_model = get_tokenizer_and_model(
            condition_config['text_condition_config']['text_embed_model'],
            device = device,
            )
        with torch.no_grad():
            empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
        logger.info(
            'Loaded text encoder: %s',
            condition_config['text_condition_config']['text_embed_model'],
            )

    im_dataset_cls = {
        'mnist'  : MnistDataset,
        'celebhq': CelebDataset,
        }.get(dataset_config['name'])
    if im_dataset_cls is None:
        raise ValueError(f'Unsupported dataset: {dataset_config["name"]}')

    task_root = Path(train_config['task_name'])
    latent_dir_name = train_config.get('vqvae_latent_dir_name')
    latent_path = resolve_task_relative_path(task_root, latent_dir_name)
    if latent_path is not None:
        logger.info('Latent path resolved to %s', latent_path)

    im_dataset = im_dataset_cls(
        split = 'train',
        im_path = dataset_config['im_path'],
        im_size = dataset_config['im_size'],
        im_channels = dataset_config['im_channels'],
        use_latents = True,
        latent_path = str(latent_path) if latent_path is not None else None,
        condition_config = condition_config,
        )

    use_latents = getattr(im_dataset, 'use_latents', False)
    train_dataset = im_dataset

    if train_imgs is not None and train_imgs > 0:
        limit = min(train_imgs, len(im_dataset))
        indices = torch.randperm(len(im_dataset))[:limit].tolist()
        train_dataset = torch.utils.data.Subset(im_dataset, indices)
        logger.info('Limiting dataset to %d images for training/debug.', limit)

    data_loader = DataLoader(
        train_dataset,
        batch_size = train_config['ldm_batch_size'],
        shuffle = True,
        )

    logger.info('Dataset: %s | Samples: %d', dataset_config['name'], len(train_dataset))
    logger.info('Batch size: %d | Epochs: %d | Learning rate: %.6f', train_config['ldm_batch_size'], train_config['ldm_epochs'], train_config['ldm_lr'])
    logger.info('Using latents: %s', use_latents)

    scheduler = LinearNoiseScheduler(
        num_timesteps = diffusion_config['num_timesteps'],
        beta_start = diffusion_config['beta_start'],
        beta_end = diffusion_config['beta_end'],
        )

    model = Unet(
        im_channels = autoencoder_model_config['z_channels'],
        model_config = diffusion_model_config,
        ).to(device)
    model.train()

    vae = None
    if not use_latents:
        logger.info('Latents not found; loading VQ-VAE for on-the-fly encoding.')
        vae = VQVAE(
            im_channels = dataset_config['im_channels'],
            model_config = autoencoder_model_config,
            ).to(device)
        vae.eval()

        vae_ckpt_value = train_config.get('vqvae_autoencoder_ckpt_name')
        vae_ckpt_path = resolve_task_relative_path(task_root, vae_ckpt_value) if vae_ckpt_value else None
        if vae_ckpt_path is not None and Path(vae_ckpt_path).exists():
            logger.info('Loaded VQ-VAE checkpoint from %s', vae_ckpt_path)
            vae.load_state_dict(torch.load(str(vae_ckpt_path), map_location = device))
        else:
            raise FileNotFoundError('VAE checkpoint not found and use_latents was disabled')

        for param in vae.parameters():
            param.requires_grad = False

    num_epochs = int(train_config['ldm_epochs'])
    optimizer = Adam(model.parameters(), lr = train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    ckpt_name = train_config.get('ldm_ckpt_name', 'ddpm_ckpt.pth')

    start_epoch_value = max(0, start_epoch) if start_epoch is not None else 0
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f'Diffusion model checkpoint not found at {resume_path}')
        logger.info('Resuming diffusion model from checkpoint: %s', resume_path)
        model.load_state_dict(torch.load(str(resume_path), map_location = device))
        inferred_epoch = infer_epoch_from_path(resume_path)
        if inferred_epoch is not None and start_epoch is None:
            start_epoch_value = inferred_epoch
        if start_epoch_value:
            logger.info('Starting training from epoch %d', start_epoch_value + 1)

    if start_epoch_value >= num_epochs:
        logger.warning(
            'Start epoch (%d) is greater than or equal to total epochs (%d); nothing to train.',
            start_epoch_value,
            num_epochs,
            )
        return

    save_every_config = train_config.get('ldm_save_every_epochs')
    resolved_save_every = save_every_epochs if save_every_epochs is not None else save_every_config
    save_every = max(1, int(resolved_save_every)) if resolved_save_every is not None else 1
    loss_history: List[Dict[str, float]] = []

    for epoch_idx in range(start_epoch_value, num_epochs):
        epoch_losses: List[float] = []

        for data in tqdm(data_loader, desc = f'Epoch {epoch_idx + 1}/{num_epochs}', leave = False):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data

            optimizer.zero_grad()
            im = im.float().to(device)

            if not use_latents:
                assert vae is not None
                with torch.no_grad():
                    im, _ = vae.encode(im)

            if 'text' in condition_types and cond_input is not None:
                assert text_tokenizer is not None and text_model is not None and empty_text_embed is not None
                assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                with torch.no_grad():
                    text_condition = get_text_representation(
                        cond_input['text'],
                        text_tokenizer,
                        text_model,
                        device,
                        )
                    text_drop_prob = get_config_value(
                        condition_config['text_condition_config'],
                        'cond_drop_prob', 0.,
                        )
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                cond_input['text'] = text_condition

            if 'image' in condition_types and cond_input is not None:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(condition_config)
                cond_input_image = cond_input['image'].to(device)
                im_drop_prob = get_config_value(
                    condition_config['image_condition_config'],
                    'cond_drop_prob', 0.,
                    )
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)

            if 'class' in condition_types and cond_input is not None:
                assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
                validate_class_config(condition_config)
                class_condition = torch.nn.functional.one_hot(
                    cond_input['class'],
                    condition_config['class_condition_config']['num_classes'],
                    ).to(device)
                class_drop_prob = get_config_value(
                    condition_config['class_condition_config'],
                    'cond_drop_prob', 0.,
                    )
                cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)

            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input = cond_input)
            loss = criterion(noise_pred, noise)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            'Epoch %d/%d | Loss: %.4f | LR: %.6f',
            epoch_idx + 1,
            num_epochs,
            avg_loss,
            current_lr,
            )

        loss_history.append({'epoch': epoch_idx + 1, 'ldm_loss': avg_loss})
        persist_loss_history(loss_history, run_artifacts.logs_dir)
        plot_epoch_loss_curve(epoch_idx, epoch_losses, run_artifacts.logs_dir)

        should_save = ((epoch_idx + 1) % save_every == 0) or (epoch_idx + 1 == num_epochs)
        if should_save:
            checkpoint_paths = save_model_checkpoint(
                model,
                ckpt_name,
                run_artifacts,
                epoch_idx,
                )
            logger.info(
                'Saved checkpoints: latest=%s | epoch=%s',
                checkpoint_paths['latest'],
                checkpoint_paths['epoch'],
                )

    logger.info('Training complete. Artifacts stored in %s', run_artifacts.run_dir)


if __name__ == '__main__':
    config_module = 'config.celebhq_params'
    output_root = 'runs'
    save_every_epochs = 5
    resume_checkpoint = None
    train_imgs = None  # e.g. 500 to debug with a subset

    train(
        config_module = config_module,
        output_root = output_root,
        save_every_epochs = save_every_epochs,
        resume_checkpoint = resume_checkpoint,
        train_imgs = train_imgs,
        )
