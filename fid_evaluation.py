"""
Lightweight FID evaluation script.

Update the variables in the ``if __name__ == '__main__':`` block to point at
your dataset, generated folder, and preferred batch size.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_frechet_distance

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def gather_image_paths(root: Path | str, limit: Optional[int]) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Image directory does not exist: {root}')
    paths = sorted(
        path for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f'No image files found under {root}')
    if limit is not None:
        paths = paths[:limit]
    return paths


def load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert('RGB')
        return transform(img)


def extract_activations(
    image_paths: Sequence[Path],
    model: torch.nn.Module,
    transform,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    features: List[np.ndarray] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc = 'Activations', leave = False):
        batch_paths = image_paths[start:start + batch_size]
        tensors = [load_image_tensor(path, transform) for path in batch_paths]
        batch = torch.stack(tensors, dim = 0).to(device)
        with torch.inference_mode():
            outputs = model(batch)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        features.append(outputs.detach().cpu().numpy())
    return np.concatenate(features, axis = 0)


def compute_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis = 0)
    sigma = np.cov(activations, rowvar = False)
    return mu.astype(np.float64), sigma.astype(np.float64)


def save_stats(path: Path, mu: np.ndarray, sigma: np.ndarray, count: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(path, mu = mu, sigma = sigma, count = count)


def load_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    data = np.load(Path(path))
    return data['mu'], data['sigma'], int(data['count'])


def calculate_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def build_inception_model() -> torch.nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights = weights, transform_input = False)
    model.fc = torch.nn.Identity()
    model.dropout = torch.nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device)


if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    DATASET_DIR = r'D:\datasets\CelebAMask-HQ\CelebAMask-HQ\256'
    GENERATED_DIR = r'FID_Images/DiT_9L'
    GENERATED_DIR = r'FID_Images/DiT_12L'
    GENERATED_DIR = r'FID_Images/Unet'
    BATCH_SIZE = 64
    MAX_DATASET = None      # e.g. 500 for a quick sanity check
    MAX_GENERATED = None
    DATASET_STATS_PATH = Path('metrics/fid/dataset_stats_256.npz')
    FORCE_DATASET_RECOMPUTE = False
    # ------------------------------------------------------------------ #

    transform = T.Compose([
        T.Resize((299, 299), interpolation = T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    inception_model = build_inception_model()

    dataset_paths = gather_image_paths(DATASET_DIR, MAX_DATASET)
    generated_paths = gather_image_paths(GENERATED_DIR, MAX_GENERATED)

    print(f'Found {len(dataset_paths)} dataset images and {len(generated_paths)} generated images.')

    if not FORCE_DATASET_RECOMPUTE and DATASET_STATS_PATH.exists():
        dataset_mu, dataset_sigma, dataset_count = load_stats(DATASET_STATS_PATH)
        print(f'Loaded dataset stats from {DATASET_STATS_PATH} ({dataset_count} samples).')
    else:
        dataset_acts = extract_activations(dataset_paths, inception_model, transform, BATCH_SIZE)
        dataset_mu, dataset_sigma = compute_statistics(dataset_acts)
        dataset_count = len(dataset_paths)
        if DATASET_STATS_PATH is not None:
            save_stats(DATASET_STATS_PATH, dataset_mu, dataset_sigma, len(dataset_paths))
            print(f'Saved dataset stats to {DATASET_STATS_PATH}')

    gen_acts = extract_activations(generated_paths, inception_model, transform, BATCH_SIZE)
    gen_mu, gen_sigma = compute_statistics(gen_acts)

    fid_value = calculate_fid(dataset_mu, dataset_sigma, gen_mu, gen_sigma)
    print('--------------------------------------------------')
    print(f'Dataset images : {dataset_count}')
    print(f'Generated imgs : {len(generated_paths)}')
    print(f'FID            : {fid_value:.4f}')
    print('--------------------------------------------------')
