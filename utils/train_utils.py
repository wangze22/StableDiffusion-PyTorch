import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


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


def persist_loss_history(loss_history: List[Dict[str, float]], logs_dir: Path) -> None:
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
        plt.plot(epochs, [entry[metric] for entry in loss_history], label = metric)
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

    steps = np.arange(1, len(losses) + 1)
    plt.figure(figsize = (10, 6))
    plt.plot(steps, losses, label = f'Epoch {epoch_idx}')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss per Step - Epoch {epoch_idx}')
    plt.grid(True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
    plt.tight_layout()
    plt.savefig(loss_dir / f'epoch_{epoch_idx:03d}.png')
    plt.close()
