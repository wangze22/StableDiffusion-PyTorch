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
import psutil
import time
from models.unet_cond_base_relu import Unet

from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from config import celebhq_text_image_cond_tc05 as cfg
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

# 量化加噪训练
from cim_qn_train.progressive_qn_train import *
import cim_layers.register_dict as reg_dict
import config.andi_config as andi_cfg

device = 'cuda'
model = Unet(im_channels = cfg.autoencoder_z_channels, model_config = cfg.diffusion_model_config).to(device)

text_tokenizer, text_model = get_tokenizer_and_model(cfg.ldm_text_condition_text_embed_model, device = device)

text_prompt = ['hello']
text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)

x_t = torch.randn(1, 4, 32, 32).to(device)
t = torch.tensor([100]).to(device)
# print(x_t.shape)
# print(text_prompt_embed.shape)
cond_input = {'text': text_prompt_embed, 'image': torch.randn([1, 18, 512, 512]).to(device)}
epsilon_theta_t = model(x_t, t, cond_input)