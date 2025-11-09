#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concatenate CSV logs from subfolders named by noise coefficients and plot loss vs. epoch.

Usage:
    python concate_csv.py

Configuration:
- Edit the CONFIG section in this script to adjust:
  - epoch_col, loss_col
  - output_csv, output_plot
  - recursive
  - y_auto, y_quantile, y_padding, y_range_mult
  - title

Behavior:
- Treat the current working directory as the root directory containing multiple
  subdirectories whose names are noise coefficients (e.g., "0.0", "0.05", "0.1").
- Read all CSV files in each noise folder (default: only files directly under it; if CONFIG['recursive']=True, include nested CSVs).
- Detect epoch and loss columns automatically (can be forced via CONFIG).
- Build a combined CSV with columns: noise, local_epoch, global_epoch, loss, source_folder, source_file, plus any original columns.
- Save the combined CSV and draw a plot of loss vs global_epoch with vertical lines marking noise changes and annotations of noise values.

Outputs (defaults, saved in the current working directory):
- concatenated_losses.csv
- loss_vs_epoch_with_noise.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Helpers ------------------------------------

def is_float_name(name: str) -> Optional[float]:
    """Return float value if the name looks like a float number, else None.
    Accepts names like '0', '0.0', '-0.1', '1e-2'.
    """
    try:
        # Strip common prefixes/suffixes (optional): not applied to keep strictness
        value = float(name)
        return value
    except Exception:
        return None


def guess_columns(df: pd.DataFrame, prefer_epoch: Optional[str] = None, prefer_loss: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Guess epoch and loss column names from the dataframe.

    Priority for epoch columns: prefer_epoch, then ['epoch', 'Epoch', 'step', 'Step', 'iter', 'Iter', 'iteration', 'Iteration', 'batch', 'Batch'].
    Priority for loss columns: prefer_loss, then ['loss', 'Loss', 'train_loss', 'Train_Loss', 'val_loss', 'Val_Loss', 'running_loss'].
    Returns (epoch_col, loss_col) which can be None if not found.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    epoch_candidates = []
    if prefer_epoch and prefer_epoch in df.columns:
        epoch_candidates.append(prefer_epoch)
    else:
        for c in ['epoch', 'step', 'iter', 'iteration', 'batch']:
            if c in cols_lower:
                epoch_candidates.append(cols_lower[c])
                break

    loss_candidates = []
    if prefer_loss and prefer_loss in df.columns:
        loss_candidates.append(prefer_loss)
    else:
        for c in ['loss', 'train_loss', 'val_loss', 'running_loss']:
            if c in cols_lower:
                loss_candidates.append(cols_lower[c])
                break

    epoch_col = epoch_candidates[0] if epoch_candidates else None
    loss_col = loss_candidates[0] if loss_candidates else None

    return epoch_col, loss_col


def ensure_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors='coerce')


# ----------------------------- Core logic ---------------------------------

def collect_noise_folders(root: Path) -> List[Tuple[float, Path]]:
    """Find subdirectories under root whose names follow the pattern 'str_noise'.
    Extract the float value after the last underscore and return
    a list of tuples (noise_value, path), sorted by noise_value asc.
    """
    folders: List[Tuple[float, Path]] = []
    for p in root.iterdir():
        if p.is_dir():
            name = p.name
            # Take the suffix after the last underscore as the numeric part.
            suffix = name.split('_')[-1] if '_' in name else name
            val = is_float_name(suffix)
            if val is not None:
                folders.append((val, p))
    folders.sort(key=lambda x: x[0])
    return folders


def find_csv_files(folder: Path, recursive: bool = False) -> List[Path]:
    """Find CSV files under a noise folder.

    Behavior:
    - If recursive=True: include all CSVs under the folder (rglob).
    - If recursive=False: include CSVs directly under the folder AND any CSVs under a
      common "logs" subdirectory (non-recursive within logs first; if none found,
      we also try recursive search inside the "logs" subdirectory only).
    This matches typical layouts like: <noise>/logs/losses.csv
    """
    files: List[Path] = []
    if recursive:
        return sorted([p for p in folder.rglob('*.csv') if p.is_file()])

    # Non-recursive: CSVs directly under the noise folder
    files.extend([p for p in folder.glob('*.csv') if p.is_file()])

    # Also look into a standard "logs" subfolder
    logs_dir = folder / 'logs'
    if logs_dir.exists() and logs_dir.is_dir():
        logs_csvs = [p for p in logs_dir.glob('*.csv') if p.is_file()]
        if not logs_csvs:
            # As a fallback, search recursively only within logs/
            logs_csvs = [p for p in logs_dir.rglob('*.csv') if p.is_file()]
        files.extend(logs_csvs)

    return sorted(files)


def read_and_extract(
    csv_path: Path,
    noise_value: float,
    prefer_epoch: Optional[str],
    prefer_loss: Optional[str],
) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Read a CSV and return a minimally processed DataFrame with essential columns.

    Adds columns: noise, source_folder, source_file. Attempts to identify epoch and loss columns.
    Returns (df, epoch_col, loss_col).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to read CSV: {csv_path} -> {e}", file=sys.stderr)
        return pd.DataFrame(), None, None

    epoch_col, loss_col = guess_columns(df, prefer_epoch, prefer_loss)

    # Add metadata
    df['noise'] = noise_value
    df['source_folder'] = str(csv_path.parent.name)
    df['source_file'] = str(csv_path.name)

    # Normalize epoch and loss columns (create if missing)
    if epoch_col is None:
        # Create a local index-based epoch (0-based)
        df['__local_epoch_from_index'] = np.arange(len(df), dtype=float)
        epoch_col = '__local_epoch_from_index'
    df[epoch_col] = ensure_numeric(df[epoch_col])

    if loss_col is None:
        # Try other numeric columns as fallback if no explicit loss column
        numeric_cols = [c for c in df.columns if c not in {epoch_col, 'noise', 'source_folder', 'source_file'}]
        found_numeric = None
        for c in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                found_numeric = c
                break
        if found_numeric is None:
            # As a last resort, create a dummy column (all NaNs)
            df['__loss_missing'] = np.nan
            loss_col = '__loss_missing'
        else:
            loss_col = found_numeric
    df[loss_col] = ensure_numeric(df[loss_col])

    return df, epoch_col, loss_col


def build_combined(
    root: Path,
    prefer_epoch: Optional[str] = None,
    prefer_loss: Optional[str] = None,
    recursive: bool = False,
) -> Tuple[pd.DataFrame, List[Tuple[float, float]], str, str]:
    """Build the combined DataFrame and return it along with segment boundaries and selected column names.

    Returns: (combined_df, segments, epoch_col_name, loss_col_name)
    - segments: list of (noise_value, start_global_epoch) for each segment (in order)
    - epoch_col_name, loss_col_name: the standardized names 'local_epoch' and 'loss' in output
    """
    noise_folders = collect_noise_folders(root)
    if not noise_folders:
        raise RuntimeError(f"No noise folders (numeric-named directories) found under: {root}")

    all_rows: List[pd.DataFrame] = []

    global_offset = 0.0
    segments: List[Tuple[float, float]] = []  # (noise_value, start_global_epoch)

    chosen_epoch_col: Optional[str] = None
    chosen_loss_col: Optional[str] = None

    for noise_value, folder in noise_folders:
        csv_files = find_csv_files(folder, recursive=recursive)
        if not csv_files:
            print(f"[WARN] No CSV files under noise folder: {folder}")
            continue

        # Concatenate within this noise to keep local order
        per_noise_frames: List[pd.DataFrame] = []
        per_noise_epoch_col: Optional[str] = None
        per_noise_loss_col: Optional[str] = None

        for csv_path in csv_files:
            df, epoch_col, loss_col = read_and_extract(csv_path, noise_value, prefer_epoch, prefer_loss)
            if df.empty:
                continue
            per_noise_frames.append(df)
            # Track the first detected columns as representative for this noise
            if per_noise_epoch_col is None and epoch_col is not None:
                per_noise_epoch_col = epoch_col
            if per_noise_loss_col is None and loss_col is not None:
                per_noise_loss_col = loss_col

        if not per_noise_frames:
            continue

        noise_df = pd.concat(per_noise_frames, ignore_index=True)

        # Resolve epoch/loss columns for this noise
        epoch_col = per_noise_epoch_col if per_noise_epoch_col is not None else '__local_epoch_from_index'
        loss_col = per_noise_loss_col if per_noise_loss_col is not None else '__loss_missing'

        # Ensure numeric
        noise_df[epoch_col] = ensure_numeric(noise_df[epoch_col])
        noise_df[loss_col] = ensure_numeric(noise_df[loss_col])

        # Sort by the epoch within noise to preserve monotonicity
        noise_df = noise_df.sort_values(by=[epoch_col, 'source_file']).reset_index(drop=True)

        # Normalize local epoch to start at 0 and be monotonic even if resets
        # If epoch within joined files resets, we enforce a monotonic sequence by detecting non-increasing steps
        local_epochs = noise_df[epoch_col].to_numpy(copy=True)
        # Handle NaNs by replacing with previous+1 or 0
        for i in range(len(local_epochs)):
            if not np.isfinite(local_epochs[i]):
                local_epochs[i] = local_epochs[i-1] + 1 if i > 0 else 0.0
            elif i > 0 and local_epochs[i] < local_epochs[i-1]:
                # If reset detected, continue from previous + 1
                local_epochs[i] = local_epochs[i-1] + 1
        # Shift so first value is 0
        if len(local_epochs) > 0:
            local_epochs = local_epochs - local_epochs[0]
        noise_df['local_epoch'] = local_epochs

        # Compute global epoch
        noise_df['global_epoch'] = noise_df['local_epoch'] + global_offset

        # Record the start point for this noise segment
        start_global = float(global_offset)
        segments.append((noise_value, start_global))

        # Prepare for next segment: increase offset by last local epoch delta
        if len(noise_df) > 0:
            global_offset = float(noise_df['global_epoch'].iloc[-1] + 1)

        # Standardize loss column name in the final output
        noise_df = noise_df.rename(columns={loss_col: 'loss'})

        # Remember chosen columns for final reporting (first non-default found)
        if chosen_epoch_col is None:
            chosen_epoch_col = 'local_epoch'
        if chosen_loss_col is None:
            chosen_loss_col = 'loss'

        all_rows.append(noise_df)

    if not all_rows:
        raise RuntimeError("No valid CSV rows were read from any noise folder.")

    combined = pd.concat(all_rows, ignore_index=True)

    # Final sorting by global epoch to ensure continuity across noise changes
    combined = combined.sort_values(by=['global_epoch', 'noise']).reset_index(drop=True)

    return combined, segments, chosen_epoch_col or 'local_epoch', chosen_loss_col or 'loss'


def plot_loss(
    combined: pd.DataFrame,
    segments: List[Tuple[float, float]],
    output_png: Path,
    title: Optional[str] = None,
    y_auto: str = 'robust',
    y_quantile: float = 0.01,
    y_padding: float = 0.03,
    y_range_mult: float = 1.0,
):
    if combined.empty:
        print("[WARN] Combined DataFrame is empty; skipping plot.")
        return

    x = combined['global_epoch'].to_numpy()
    y = ensure_numeric(combined['loss']).to_numpy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y, label='loss', color='tab:blue', linewidth=1.5)

    # X-range info for placing vertical labels
    if len(x) > 0 and np.isfinite(np.nanmin(x)) and np.isfinite(np.nanmax(x)):
        x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        x_range = max(x_max - x_min, 1.0)
    else:
        x_min, x_max, x_range = 0.0, 1.0, 1.0

    # Compute Y limits according to auto-scaling strategy
    finite_mask = np.isfinite(y)
    if not np.any(finite_mask):
        y_min, y_max = 0.0, 1.0
    else:
        y_f = y[finite_mask]
        if y_auto == 'full':
            y_min, y_max = float(np.nanmin(y_f)), float(np.nanmax(y_f))
        elif y_auto == 'tight':
            y_min, y_max = float(np.nanmin(y_f)), float(np.nanmax(y_f))
        else:  # 'robust'
            q = min(max(y_quantile, 0.0), 0.25)
            y_min = float(np.nanquantile(y_f, q))
            y_max = float(np.nanquantile(y_f, 1 - q))
            # report clipped points
            clipped_low = int((y < y_min).sum())
            clipped_high = int((y > y_max).sum())
            clipped = clipped_low + clipped_high
            if clipped > 0:
                total = int(np.isfinite(y).sum())
                print(f"[INFO] Robust y-limit excluded {clipped}/{total} points (low={clipped_low}, high={clipped_high}) using q={q:.3f}")
        # Ensure non-zero range
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            y_min = float(np.nanmin(y_f))
            y_max = float(np.nanmax(y_f))
            if y_min == y_max:
                eps = max(abs(y_min) * 0.05, 1e-4)
                y_min -= eps
                y_max += eps

    # Base range from selected auto-scaling
    base_range = max(y_max - y_min, 1e-12)

    # Scale range strength: 1.0 means exactly base loss range; n means n times that range, centered.
    m = max(float(y_range_mult), 1.0)
    scaled_range = max(base_range * m, 1e-12)
    center = 0.5 * (y_min + y_max)
    y_min_s = center - 0.5 * scaled_range
    y_max_s = center + 0.5 * scaled_range

    # Apply padding for readability and for red labels visibility (relative to scaled range)
    pad = y_padding * scaled_range
    ax.set_ylim(y_min_s - pad, y_max_s + pad)

    dx = 0.02 * x_range  # horizontal offset to move label away from the line
    dy_top = y_padding * scaled_range  # vertical margin from the top

    for idx, (noise_value, start_x) in enumerate(segments):
        if idx == 0:
            continue
        ax.axvline(x=start_x, color='tab:red', linestyle='--', alpha=0.6, zorder=3)
        # y coordinate slightly below the current top after plotting
        y_lo, y_hi = ax.get_ylim()
        y_pos = y_hi - dy_top
        # keep label within right boundary
        x_lo, x_hi = ax.get_xlim()
        x_pos = min(start_x + dx, x_hi - 0.01 * (x_hi - x_lo))
        ax.text(
            x_pos,
            y_pos,
            f" noise→{noise_value} ",
            color='tab:red',
            va='top',
            ha='left',
            fontsize=9,
            rotation=90,
            alpha=0.9,
            zorder=4,
            clip_on=False,
        )

    # Removed black mid-segment noise annotations as requested

    ax.set_xlabel('Global Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title or 'Loss vs Epoch with Noise Changes')
    ax.grid(True, linestyle='--', alpha=0.3)

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


# ----------------------------- CONFIG ----------------------------------------

# Edit the values below to adjust behavior (no argparse/CLI is used)
CONFIG: Dict[str, object] = {
    'epoch_col': None,                 # e.g., 'epoch' to force epoch column
    'loss_col': None,                  # e.g., 'loss' to force loss column
    'output_csv': 'concatenated_losses.csv',
    'output_plot': 'loss_vs_epoch_with_noise.png',
    'recursive': False,                # True to search CSVs recursively in each noise folder
    # Y-axis auto-scaling and layout
    'y_auto': 'robust',                # 'robust' | 'full' | 'tight'
    'y_quantile': 0.01,                # only used in 'robust' mode
    'y_padding': 0.03,                 # visual padding relative to final y-range
    'y_range_mult': 5.0,               # n>=1.0: final window is n× base loss range (centered)
    # Plot title (None -> auto)
    'title': None,
}


def main(argv: Optional[List[str]] = None) -> int:
    # Read settings from CONFIG
    epoch_col = CONFIG.get('epoch_col')  # type: ignore[assignment]
    loss_col = CONFIG.get('loss_col')    # type: ignore[assignment]
    output_csv = str(CONFIG.get('output_csv') or 'concatenated_losses.csv')
    output_plot = str(CONFIG.get('output_plot') or 'loss_vs_epoch_with_noise.png')
    recursive = bool(CONFIG.get('recursive'))

    y_auto = str(CONFIG.get('y_auto') or 'robust')
    y_quantile = float(CONFIG.get('y_quantile') or 0.01)
    y_padding = float(CONFIG.get('y_padding') or 0.03)
    y_range_mult = float(CONFIG.get('y_range_mult') or 1.0)
    title = CONFIG.get('title')

    root = Path.cwd()  # Current directory as the root containing noise folders

    try:
        combined, segments, epoch_col_name, loss_col_name = build_combined(
            root,
            prefer_epoch=epoch_col if isinstance(epoch_col, str) else None,
            prefer_loss=loss_col if isinstance(loss_col, str) else None,
            recursive=recursive,
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Save combined CSV
    out_csv_path = (root / output_csv) if not Path(output_csv).is_absolute() else Path(output_csv)
    try:
        combined.to_csv(out_csv_path, index=False)
        print(f"[INFO] Saved combined CSV: {out_csv_path}")
    except Exception as e:
        print(f"[WARN] Failed to save combined CSV to {out_csv_path}: {e}", file=sys.stderr)

    # Plot
    out_png_path = (root / output_plot) if not Path(output_plot).is_absolute() else Path(output_plot)
    try:
        plot_loss(
            combined,
            segments,
            out_png_path,
            title=(title if isinstance(title, str) and title else f'Loss vs Epoch (Noise segments = {len(segments)})'),
            y_auto=y_auto,
            y_quantile=y_quantile,
            y_padding=y_padding,
            y_range_mult=y_range_mult,
        )
        print(f"[INFO] Saved plot: {out_png_path}")
    except Exception as e:
        print(f"[WARN] Failed to save plot to {out_png_path}: {e}", file=sys.stderr)

    # Simple summary
    try:
        summary = combined.groupby('noise', as_index=False).agg(rows=('loss', 'size'), start=('global_epoch', 'min'), end=('global_epoch', 'max'))
        print("[INFO] Segment summary:")
        for _, r in summary.iterrows():
            print(f"  noise={r['noise']}: rows={int(r['rows'])}, global_epoch=[{r['start']:.1f}, {r['end']:.1f}]")
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
