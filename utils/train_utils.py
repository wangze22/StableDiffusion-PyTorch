import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

# Force a non-interactive backend so matplotlib does not rely on Tk in worker processes.
matplotlib.use('Agg', force = True)
import matplotlib.pyplot as plt
import numpy as np
import json
import inspect
import torch


def ensure_directory(path: Path) -> None:
    """Create directory tree if it does not exist."""
    path.mkdir(parents = True, exist_ok = True)


def create_run_artifacts(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare run/checkpoint/log directories and logger."""
    output_root = train_config.get('ldm_output_root', 'runs')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    task_name = train_config.get('task_name', 'ddpm')

    run_dir = Path(output_root) / f'ddpm_{timestamp}' / task_name
    checkpoints_dir = run_dir / 'checkpoints'
    logs_dir = run_dir / 'logs'

    for path in (checkpoints_dir, logs_dir):
        ensure_directory(path)

    logger_name = f'scripts_refined_ddpm_{timestamp}'
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
        'run_dir': run_dir,
        'checkpoints_dir': checkpoints_dir,
        'logs_dir': logs_dir,
        'logger': logger,
    }


def save_config_snapshot_json(logs_dir: Path, cfg_module: Any) -> Path:
    """Serialize the given config module into a JSON file under logs_dir.

    The snapshot includes two top-level keys:
      - meta: module name and source file path
      - config: all public (non-underscore) attributes converted to JSON-friendly forms

    Returns the path to the written JSON file.
    """
    def _to_jsonable(obj):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple, set)):
            return [_to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in obj.items()}
        # Torch device/dtype
        if isinstance(obj, torch.device):
            return str(obj)
        if hasattr(torch, 'dtype') and isinstance(obj, torch.dtype):
            return str(obj)
        # Fallback: string representation
        return str(obj)

    cfg_items: Dict[str, Any] = {}
    for name in dir(cfg_module):
        if name.startswith('_'):
            continue
        try:
            val = getattr(cfg_module, name)
        except Exception:
            continue
        if inspect.ismodule(val) or inspect.isfunction(val) or inspect.ismethod(val):
            continue
        cfg_items[name] = _to_jsonable(val)

    cfg_meta = {
        'config_module': getattr(cfg_module, '__name__', 'unknown'),
        'config_file': getattr(cfg_module, '__file__', None),
    }

    snapshot_path = Path(logs_dir) / 'config_snapshot.json'
    with snapshot_path.open('w', encoding='utf-8') as snapshot_file:
        json.dump({'meta': cfg_meta, 'config': cfg_items}, snapshot_file, ensure_ascii=False, indent=2)
    return snapshot_path


def persist_loss_history(loss_history: List[Dict[str, float]], logs_dir: Path, smoothing_alpha: Optional[float] = None) -> None:
    """Write loss history to CSV and save aggregate plot."""
    if not loss_history:
        return

    csv_path = Path(logs_dir) / 'losses.csv'
    fieldnames = list(loss_history[0].keys())

    with csv_path.open('w', newline = '') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)

    epochs = [entry['epoch'] for entry in loss_history]
    metrics = [field for field in fieldnames if field != 'epoch']
    if not metrics:
        return

    plt.figure(figsize = (10, 6))
    for metric in metrics:
        values = [entry[metric] for entry in loss_history]
        plt.plot(epochs, values, label = metric)
        if smoothing_alpha is not None and values:
            smoothed_values = []
            for value in values:
                if not smoothed_values:
                    smoothed_values.append(value)
                else:
                    prev = smoothed_values[-1]
                    smoothed_values.append((1.0 - smoothing_alpha) * prev + smoothing_alpha * value)
            plt.plot(
                epochs,
                smoothed_values,
                label = f'{metric} EMA alpha={smoothing_alpha:.2f}',
                linewidth = 2.0,
            )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDPM Training Losses')
    plt.legend()
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(Path(logs_dir) / 'loss_curve.png')
    plt.close()
def plot_epoch_loss_curve(epoch_idx: int, losses: List[float], logs_dir: Path) -> None:
    """Plot per-step loss trend for a single epoch."""
    if not losses:
        return

    loss_dir = Path(logs_dir) / 'epoch_loss_plots'
    ensure_directory(loss_dir)

    raw_losses = np.asarray(losses, dtype = np.float64)
    steps = np.arange(1, len(losses) + 1)

    plt.figure(figsize = (10, 6))
    plt.plot(steps, raw_losses, label = f'Epoch {epoch_idx}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss per Step - Epoch {epoch_idx}')
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(loss_dir / f'epoch_{epoch_idx:03d}.png')
    plt.close()
