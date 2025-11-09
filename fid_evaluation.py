"""
FID evaluation helper with two usage modes:

1. Precompute dataset statistics (Inception activations) once and dump them to disk.
2. Reuse cached statistics to score any new batch of generated images without
   reprocessing the dataset.

All configuration happens inside the ``if __name__ == '__main__':`` block.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def build_transform():
    return T.Compose([
        T.Resize((299, 299), interpolation = T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights = weights, aux_logits = False, transform_input = False)
        model.fc = torch.nn.Identity()
        model.dropout = torch.nn.Identity()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model.to(device)

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)


def list_dataset_images(cfg, max_items: Optional[int]) -> List[Path]:
    if cfg.condition_config is None:
        raise ValueError('condition_config must be defined inside the config module.')
    dataset = CelebDataset(
        split = 'train',
        im_path = cfg.dataset_im_path,
        im_size = cfg.dataset_im_size,
        im_channels = cfg.dataset_im_channels,
        use_latents = False,
        latent_path = None,
        condition_config = cfg.condition_config,
    )
    paths = [Path(p) for p in dataset.images]
    if max_items is not None:
        paths = paths[:max_items]
    return paths


def list_generated_images(root: Path, max_items: Optional[int]) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f'Generated image directory does not exist: {root}')
    candidates = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not candidates:
        raise ValueError(f'No image files found under {root}')
    if max_items is not None:
        candidates = candidates[:max_items]
    return candidates


def load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert('RGB')
        tensor = transform(img)
    return tensor


def compute_activations(
    image_paths: Sequence[Path],
    extractor: InceptionFeatureExtractor,
    batch_size: int,
    transform,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError('batch_size must be >= 1')
    features: List[np.ndarray] = []
    iterator = range(0, len(image_paths), batch_size)
    for start in tqdm(iterator, desc = 'Activations', leave = False):
        batch_paths = image_paths[start:start + batch_size]
        tensors = [load_image_tensor(path, transform) for path in batch_paths]
        batch = torch.stack(tensors, dim = 0).to(device)
        with torch.inference_mode():
            feat = extractor(batch)
        features.append(feat.detach().cpu().numpy())
    return np.concatenate(features, axis = 0)


def compute_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(activations, axis = 0)
    sigma = np.cov(activations, rowvar = False)
    return mu.astype(np.float64), sigma.astype(np.float64)


def save_stats(path: Path, mu: np.ndarray, sigma: np.ndarray, count: int) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(path, mu = mu, sigma = sigma, count = count)


def load_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f'FID stats file not found: {path}')
    data = np.load(path)
    return data['mu'], data['sigma'], int(data['count'])


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) * 0.5


def _sqrt_and_inv_sqrt(matrix: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    matrix = _symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, eps, None)
    sqrt_vals = np.sqrt(eigvals)
    inv_sqrt_vals = np.divide(1.0, sqrt_vals, out = np.zeros_like(sqrt_vals), where = sqrt_vals > eps)
    sqrt_mat = (eigvecs * sqrt_vals) @ eigvecs.T
    inv_sqrt_mat = (eigvecs * inv_sqrt_vals) @ eigvecs.T
    return sqrt_mat, inv_sqrt_mat


def _sqrtm_product(sigma1: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    sqrt_sigma1, inv_sqrt_sigma1 = _sqrt_and_inv_sqrt(sigma1)
    middle = inv_sqrt_sigma1 @ sigma2 @ inv_sqrt_sigma1
    middle = _symmetrize(middle)
    sqrt_middle, _ = _sqrt_and_inv_sqrt(middle)
    return sqrt_sigma1 @ sqrt_middle @ sqrt_sigma1


def calculate_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm_product(sigma1, sigma2)
    trace_term = np.trace(sigma1 + sigma2 - 2.0 * covmean)
    fid = diff.dot(diff) + trace_term
    return float(np.real(fid))


def maybe_prepare_dataset_stats(
    stats_path: Path,
    cfg,
    batch_size: int,
    transform,
    force_recompute: bool,
    max_items: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int]:
    if stats_path.exists() and not force_recompute:
        mu, sigma, count = load_stats(stats_path)
        print(f'Loaded cached dataset stats from {stats_path} ({count} samples).')
        return mu, sigma, count

    print('Computing dataset activations...')
    dataset_paths = list_dataset_images(cfg, max_items)
    extractor = InceptionFeatureExtractor()
    activations = compute_activations(dataset_paths, extractor, batch_size, transform)
    mu, sigma = compute_statistics(activations)
    save_stats(stats_path, mu, sigma, len(dataset_paths))
    print(f'Dataset stats saved to {stats_path}')
    return mu, sigma, len(dataset_paths)


def main(
    cfg,
    generated_dir: Path,
    stats_path: Path,
    batch_size: int,
    force_dataset_recompute: bool,
    max_dataset_items: Optional[int],
    max_generated_items: Optional[int],
) -> None:
    transform = build_transform()
    dataset_mu, dataset_sigma, dataset_count = maybe_prepare_dataset_stats(
        stats_path = stats_path,
        cfg = cfg,
        batch_size = batch_size,
        transform = transform,
        force_recompute = force_dataset_recompute,
        max_items = max_dataset_items,
    )

    gen_paths = list_generated_images(generated_dir, max_generated_items)
    print(f'Computing activations for {len(gen_paths)} generated images...')
    extractor = InceptionFeatureExtractor()
    gen_activations = compute_activations(gen_paths, extractor, batch_size, transform)
    gen_mu, gen_sigma = compute_statistics(gen_activations)

    fid_value = calculate_fid(dataset_mu, dataset_sigma, gen_mu, gen_sigma)
    print('--------------------------------------------------')
    print(f'Dataset images : {dataset_count}')
    print(f'Generated imgs : {len(gen_paths)}')
    print(f'FID            : {fid_value:.4f}')
    print('--------------------------------------------------')


if __name__ == '__main__':
    # -------------------------------------------------------------- #
    CONFIG_MODULE = 'Model_DiT_9L_config'
    GENERATED_IMAGE_DIR = 'generated/batch_conditions'
    DATASET_STATS_PATH = 'metrics/fid/celebhq_fid_stats.npz'

    BATCH_SIZE = 32
    FORCE_DATASET_RECOMPUTE = False
    MAX_DATASET_ITEMS = None  # e.g. 500 for quick tests
    MAX_GENERATED_ITEMS = None
    # -------------------------------------------------------------- #

    cfg_module = importlib.import_module(CONFIG_MODULE)
    main(
        cfg = cfg_module,
        generated_dir = Path(GENERATED_IMAGE_DIR),
        stats_path = Path(DATASET_STATS_PATH),
        batch_size = BATCH_SIZE,
        force_dataset_recompute = FORCE_DATASET_RECOMPUTE,
        max_dataset_items = MAX_DATASET_ITEMS,
        max_generated_items = MAX_GENERATED_ITEMS,
    )
