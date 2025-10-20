import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_cond_base import Unet
from models.vqvae import VQVAE

from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from utils.config_utils import *
from utils.diffusion_utils import *
from utils.text_utils import *
from utils.train_utils import (
    create_run_artifacts,
    ensure_directory,
    persist_loss_history,
    plot_epoch_loss_curve,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(config_path):
    # Read the config file #
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    ########################

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    model_pth_config = config['model_paths']
    run_artifacts = create_run_artifacts(train_config)
    logger: logging.Logger = run_artifacts['logger']
    logger.info('Loaded config from %s', config_path)
    logger.info('Run artifacts directory: %s', run_artifacts['run_dir'])

    save_every = max(1, int(train_config.get('ldm_save_every_epochs', 1)))
    loss_history: List[Dict[str, float]] = []
    legacy_ckpt_dir = Path(train_config['task_name'])
    ensure_directory(legacy_ckpt_dir)

    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(
        num_timesteps = diffusion_config['num_timesteps'],
        beta_start = diffusion_config['beta_start'],
        beta_end = diffusion_config['beta_end'],
        )
    ###############################################

    # Instantiate Condition related components
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key = 'condition_config', default_value = None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'text' in condition_types:
            validate_text_config(condition_config)
            with torch.no_grad():
                # Load tokenizer and text model based on config
                # Also get empty text representation
                text_tokenizer, text_model = get_tokenizer_and_model(
                    condition_config['text_condition_config']
                    ['text_embed_model'], device = device,
                    )
                empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)

    im_dataset_cls = {
        'celebhq': CelebDataset,
        }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(
        split = 'train',
        im_path = dataset_config['im_path'],
        im_size = dataset_config['im_size'],
        im_channels = dataset_config['im_channels'],
        use_latents = True,
        latent_path = os.path.join(
            train_config['task_name'],
            train_config['vqvae_latent_dir_name'],
            ),
        condition_config = condition_config,
        )

    data_loader = DataLoader(
        im_dataset,
        batch_size = train_config['ldm_batch_size'],
        shuffle = True,
        )

    # Instantiate the unet model
    model = Unet(
        im_channels = autoencoder_model_config['z_channels'],
        model_config = diffusion_model_config,
        ).to(device)
    resume_path = model_pth_config['ldm_ckpt_resume']
    if resume_path is not None:
        model.load_state_dict(torch.load(str(resume_path), map_location = device))
        logger.info(f'Loaded ldm model {resume_path}')
    model.train()

    vae = None
    # Load VAE ONLY if latents are not to be saved or some are missing
    if not im_dataset.use_latents:
        logger.info('Loading vqvae model as latents not present')
        vae = VQVAE(
            im_channels = dataset_config['im_channels'],
            model_config = autoencoder_model_config,
            ).to(device)
        vae.eval()
        # Load vae if found
        if os.path.exists(
                os.path.join(
                    train_config['task_name'],
                    model_pth_config['vqvae_autoencoder_ckpt_name'],
                    ),
                ):
            logger.info('Loaded vae checkpoint')
            vae.load_state_dict(
                torch.load(
                    os.path.join(
                        train_config['task_name'],
                        model_pth_config['vqvae_autoencoder_ckpt_name'],
                        ),
                    map_location = device,
                    ),
                )
        else:
            raise Exception('VAE checkpoint not found and use_latents was disabled')

    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr = train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()

    # Load vae and freeze parameters ONLY if latents already not saved
    if not im_dataset.use_latents:
        assert vae is not None
        for param in vae.parameters():
            param.requires_grad = False

    # Run training
    for epoch_idx in range(num_epochs):
        epoch_losses: List[float] = []
        for data in tqdm(data_loader, desc = f'Epoch {epoch_idx + 1}/{num_epochs}', leave = False):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            if not im_dataset.use_latents:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            ########### Handling Conditional Input ###########
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
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
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(condition_config)
                cond_input_image = cond_input['image'].to(device)
                # Drop condition
                im_drop_prob = get_config_value(
                    condition_config['image_condition_config'],
                    'cond_drop_prob', 0.,
                    )
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)

            ################################################

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input = cond_input)
            loss = criterion(noise_pred, noise)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info(
            'Epoch %d/%d | Loss: %.4f',
            epoch_idx + 1,
            num_epochs,
            avg_loss,
            )

        loss_history.append({'epoch': epoch_idx + 1, 'ldm_loss': avg_loss})
        persist_loss_history(loss_history, run_artifacts['logs_dir'])
        plot_epoch_loss_curve(epoch_idx + 1, epoch_losses, run_artifacts['logs_dir'])

        should_save = ((epoch_idx + 1) % save_every == 0) or (epoch_idx + 1 == num_epochs)
        if should_save:
            state_dict = model.state_dict()
            checkpoints_dir = run_artifacts['checkpoints_dir']
            latest_ckpt_path = checkpoints_dir / model_pth_config['ldm_ckpt_name']
            epoch_ckpt_path = checkpoints_dir / f'epoch_{epoch_idx + 1:03d}_{model_pth_config["ldm_ckpt_name"]}'
            torch.save(state_dict, latest_ckpt_path)
            torch.save(state_dict, epoch_ckpt_path)
            legacy_ckpt_path = legacy_ckpt_dir / model_pth_config['ldm_ckpt_name']
            torch.save(state_dict, legacy_ckpt_path)
            logger.info(
                'Saved checkpoints: latest=%s | epoch=%s',
                latest_ckpt_path,
                epoch_ckpt_path,
                )

    logger.info('Training complete. Artifacts stored in %s', run_artifacts['run_dir'])
    print('Done Training ...')


if __name__ == '__main__':
    config_path = 'config/celebhq_text_image_cond.yaml'
    train(config_path)
