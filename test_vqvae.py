import os
import random
import yaml
import torch
import numpy as np
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from models.vqvae import VQVAE
from dataset.celeb_dataset import CelebDataset


# Inline configuration (no argparse)
CONFIG_PATH = 'config/celebhq.yaml'  # YAML used in training
SPLIT = 'train'                      # dataset split to visualize
NUM_COLS = 4                         # images per row; AB has 2 rows
NUM_GRIDS = 1                        # how many AB grids to save
OUTPUT_DIR = None                    # if None, will use <task_name>/vqvae_test
SEED = None                          # if None, will use train_cfg.seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise
    return cfg


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(dataset_config, split: str):
    # Only CelebHQ is supported per request (no MNIST)
    return CelebDataset(
        split=split,
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
    )


def build_model(dataset_config, autoencoder_config, ckpt_path: str | None):
    model = VQVAE(
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_config,
    ).to(device)
    model.eval()
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded VQVAE checkpoint: {ckpt_path}")
    else:
        print("Warning: checkpoint not found. Using randomly initialized weights.")
    return model


def save_ab_grid(orig: torch.Tensor, recon: torch.Tensor, out_path: str, ncol: int):
    # Expect orig and recon in [0,1]
    # Stack so that the first row (A) is originals and second row (B) is reconstructions
    grid = make_grid(torch.cat([orig, recon], dim=0), nrow=ncol)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save(out_path)
    img.close()


def main():
    # Load training config from YAML used during training
    cfg = load_config(CONFIG_PATH)
    dataset_cfg = cfg['dataset_params']
    ae_cfg = cfg['autoencoder_params']
    train_cfg = cfg['train_params']

    # Seed
    seed_val = SEED if SEED is not None else train_cfg.get('seed', 42)
    set_seed(seed_val)

    # Resolve checkpoint path using training script convention
    ckpt_path = os.path.join('tools',train_cfg['task_name'], train_cfg['vqvae_autoencoder_ckpt_name'])

    # Resolve output directory
    output_dir = OUTPUT_DIR if OUTPUT_DIR is not None else os.path.join(train_cfg['task_name'], 'vqvae_test')
    os.makedirs(output_dir, exist_ok=True)

    # Build model and dataset (CelebHQ only)
    model = build_model(dataset_cfg, ae_cfg, ckpt_path)
    dataset = build_dataset(dataset_cfg, split=SPLIT)

    loader = DataLoader(dataset, batch_size=NUM_COLS, shuffle=True)

    saved = 0
    with torch.no_grad():
        for batch in loader:
            # Some datasets may return tuple (image, label); handle gracefully
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.float().to(device)

            # Forward
            output, _, _ = model(imgs)

            # Clamp to [-1,1] then map to [0,1]
            out_vis = torch.clamp(output, -1.0, 1.0)
            out_vis = (out_vis + 1.0) / 2.0
            in_vis = (imgs + 1.0) / 2.0

            # Ensure we have exactly NUM_COLS
            n = min(NUM_COLS, in_vis.shape[0])
            in_vis = in_vis[:n].detach().cpu()
            out_vis = out_vis[:n].detach().cpu()

            out_path = os.path.join(output_dir, f'ab_compare_{saved:03d}.png')
            save_ab_grid(in_vis, out_vis, out_path, ncol=n)
            print(f'Saved: {out_path}')

            saved += 1
            if saved >= NUM_GRIDS:
                break

    print('Done. First row = A (original), Second row = B (reconstruction).')


if __name__ == '__main__':
    main()
