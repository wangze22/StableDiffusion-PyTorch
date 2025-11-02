import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
matplotlib.use('Agg', force = True)

def generate_losses(num_epochs: int = 120, seed: int = 42) -> np.ndarray:
    """Generate a synthetic loss curve that decreases with noise and rare low outliers."""
    rng = np.random.default_rng(seed)

    baseline = np.linspace(1.0, 0.15, num_epochs)
    noise = rng.normal(loc=0.0, scale=0.02, size=num_epochs)
    losses = baseline + noise

    outlier_epochs = [25, 60, 95]
    for idx in outlier_epochs:
        if 0 <= idx < num_epochs:
            losses[idx] *= 0.6
    return np.maximum(losses, 0.01)


def exponential_smoothing(values: np.ndarray, alpha: float) -> np.ndarray:
    """EMA smoothing aligned with the training loop behaviour."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = (1.0 - alpha) * smoothed[i - 1] + alpha * values[i]
    return smoothed


def main() -> None:
    num_epochs = 120
    alpha = 0.2

    raw_losses = generate_losses(num_epochs)
    smoothed_losses = exponential_smoothing(raw_losses, alpha)

    model_param = torch.nn.Parameter(torch.tensor([0.0]))
    optimizer = SGD([model_param], lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-4,
    )

    lrs = []
    for epoch_idx in range(num_epochs):
        scheduler.step(float(smoothed_losses[epoch_idx]))
        lrs.append(optimizer.param_groups[0]["lr"])

    report_lines = [
        f"{'Epoch':>5} | {'Raw Loss':>8} | {'Smoothed':>8} | {'LR':>10}",
        "-" * 45,
    ]
    for epoch_idx in range(num_epochs):
        if epoch_idx % 5 == 0 or raw_losses[epoch_idx] != smoothed_losses[epoch_idx]:
            report_lines.append(
                f"{epoch_idx + 1:5d} | {raw_losses[epoch_idx]:8.4f} | "
                f"{smoothed_losses[epoch_idx]:8.4f} | {lrs[epoch_idx]:10.6e}"
            )

    print("\n".join(report_lines))

    output_path = Path("lr_scheduler_smoothing_demo.png")
    plt.figure(figsize=(10, 6))
    plt.plot(raw_losses, label="raw loss", alpha=0.7)
    plt.plot(smoothed_losses, label="smoothed loss (EMA)", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EMA-smoothing effect on ReduceLROnPlateau input")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"\nSaved plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
