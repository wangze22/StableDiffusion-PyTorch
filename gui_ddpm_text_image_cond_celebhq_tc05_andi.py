import threading
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import random
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

import torch
import torchvision
from torchvision.utils import make_grid

from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler, DDIMSampler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from dataset.celeb_dataset import CelebDataset

import config.celebhq_text_image_cond_tc05 as cfg

from cim_qn_train.progressive_qn_train import *
import cim_layers.register_dict as reg_dict
import config.andi_config as andi_cfg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------- Labels and palette -------------
label_list: List[str] = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

# Background id 0 is black; class ids 1..18 will use the following distinct colors
palette: List[Tuple[int, int, int]] = [
    (0, 0, 0),  # background 0
    (255, 205, 148),  # skin
    (255, 255, 0),  # nose
    (0, 255, 255),  # eye_g
    (0, 128, 255),  # l_eye
    (0, 0, 255),  # r_eye
    (139, 69, 19),  # l_brow
    (160, 82, 45),  # r_brow
    (255, 105, 180),  # l_ear
    (255, 20, 147),  # r_ear
    (255, 0, 0),  # mouth
    (255, 140, 0),  # u_lip
    (178, 34, 34),  # l_lip
    (50, 50, 50),  # hair (dark gray) — distinguishable from background
    (128, 0, 128),  # hat
    (255, 0, 255),  # ear_r (ear ring)
    (173, 216, 230),  # neck_l
    (152, 251, 152),  # neck
    (0, 128, 0),  # cloth
    ]

# Ensure palette has one color per class id including background
assert len(palette) == (1 + len(label_list))


# ------------- Utility conversion functions -------------

def class_map_to_rgb(class_map: np.ndarray) -> Image.Image:
    """Convert HxW class id map (0..18) to RGB PIL image using palette."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype = np.uint8)
    for cid, color in enumerate(palette):
        rgb[class_map == cid] = color
    return Image.fromarray(rgb, mode = 'RGB')


def one_hot_from_class_map(class_map: np.ndarray, num_classes: int) -> torch.Tensor:
    """Create CxHxW one-hot tensor from HxW class ids (0..num_classes). Channel 0 corresponds to class id 1."""
    h, w = class_map.shape
    oh = np.zeros((num_classes, h, w), dtype = np.float32)
    for idx in range(num_classes):
        # class id for this channel is idx+1
        oh[idx] = (class_map == (idx + 1))
    return torch.from_numpy(oh)


def class_map_from_one_hot(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convert CxHxW one-hot mask to HxW class ids (0..C) where 0 is background."""
    # mask_tensor: CxHxW, values in {0,1}
    class_idx = torch.argmax(mask_tensor, dim = 0)  # 0..C-1
    class_map = class_idx.to(dtype = torch.int32).cpu().numpy() + 1  # 1..C
    # background: positions where all zeros remain 0
    bg = (mask_tensor.sum(dim = 0) == 0)
    class_map[bg.cpu().numpy()] = 0
    return class_map


# ------------- Diffusion sampling (adapted) -------------

def sample_with_mask_and_prompt(
        model: Unet,
        vae: VQVAE,
        text_tokenizer,
        text_model,
        mask_oh: torch.Tensor,
        prompt_text: str,
        cf_guidance_scale: float = 1.0,
        num_inference_steps: int = 100,
        method: str = 'quadratic',
        eta: float = 0.0,
        ) -> Image.Image:

    sampler = DDIMSampler(
        beta = (cfg.diffusion_beta_start, cfg.diffusion_beta_end),
        model = model,
        T = cfg.diffusion_num_timesteps,
        )

    """
    Run DDPM sampling using provided mask (1xCxHxW) and prompt text, return final decoded PIL image.
    """
    im_size = cfg.dataset_im_size // 2 ** sum(cfg.autoencoder_down_sample)

    # Random latent
    xt = torch.randn((1, cfg.autoencoder_z_channels, im_size, im_size), device = device)

    # Text embeddings
    text_prompt = [prompt_text]
    empty_prompt = ['']
    with torch.no_grad():
        text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
        empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)

    # Prepare cond inputs
    cond_input = {'text': text_prompt_embed, 'image': mask_oh.to(device)}
    uncond_input = {'text': empty_text_embed, 'image': torch.zeros_like(mask_oh).to(device)}
    num_inference_steps = max(2, min(num_inference_steps, cfg.diffusion_num_timesteps - 1))

    T_list = torch.linspace(cfg.diffusion_num_timesteps - 1, 0, num_inference_steps, dtype = torch.long)
    assert T_list[-1] == 0, 'Last timestep must be zero.'
    # Sampling loop
    with torch.no_grad():
        xt = sampler.forward(
            xt,
            cond_input,
            uncond_input,
            steps=num_inference_steps,
            method=method,
            eta=eta,
        )
        # Decode final latent
        ims = vae.decode(xt)
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow = 1)
        img = torchvision.transforms.ToPILImage()(grid)
        return img


# ------------- Model bundle -------------

@dataclass
class ModelBundle:
    model: Unet
    vae: VQVAE
    text_tokenizer: Any
    text_model: Any


def load_models_and_configs(
        model,
        vqvae_ckpt_path: Path,
        ) -> ModelBundle:
    # Validate
    assert cfg.condition_config is not None, 'Condition config required for text+image conditioning.'
    condition_types = cfg.ldm_condition_types
    assert 'text' in condition_types and 'image' in condition_types, 'Both text and image conditions are required.'

    # Tokenizer and text model
    with torch.no_grad():
        text_tokenizer, text_model = get_tokenizer_and_model(cfg.ldm_text_condition_text_embed_model, device = device)


    # VQVAE
    autoencoder_config = {
        'z_channels'     : cfg.autoencoder_z_channels,
        'codebook_size'  : cfg.autoencoder_codebook_size,
        'down_channels'  : list(cfg.autoencoder_down_channels),
        'mid_channels'   : list(cfg.autoencoder_mid_channels),
        'down_sample'    : list(cfg.autoencoder_down_sample),
        'attn_down'      : list(cfg.autoencoder_attn_down),
        'norm_channels'  : cfg.autoencoder_norm_channels,
        'num_heads'      : cfg.autoencoder_num_heads,
        'num_down_layers': cfg.autoencoder_num_down_layers,
        'num_mid_layers' : cfg.autoencoder_num_mid_layers,
        'num_up_layers'  : cfg.autoencoder_num_up_layers,
        }
    vae = VQVAE(im_channels = cfg.dataset_im_channels, model_config = autoencoder_config).to(device)
    vae.eval()
    if not vqvae_ckpt_path.exists():
        raise FileNotFoundError(f'VAE checkpoint not found: {vqvae_ckpt_path}')
    vae.load_state_dict(torch.load(str(vqvae_ckpt_path), map_location = device))

    return ModelBundle(
        model = model, vae = vae,
        text_tokenizer = text_tokenizer, text_model = text_model,
        )


# ------------- GUI -------------

class MaskPainterGUI:
    def __init__(self, master: tk.Tk, bundle: ModelBundle):
        self.master = master
        self.bundle = bundle

        # Initialize dataset once for caption alignment and efficiency
        try:
            self.dataset = CelebDataset(
                split = 'train',
                im_path = cfg.dataset_im_path,
                im_size = cfg.dataset_im_size,
                im_channels = cfg.dataset_im_channels,
                use_latents = True,
                latent_path = str(Path(cfg.train_task_name) / cfg.train_vqvae_latent_dir_name),
                condition_config = cfg.condition_config,
                )
        except Exception:
            self.dataset = None

        # Track the currently loaded sample index (for Random Prompt)
        self.current_index: Optional[int] = None

        self.num_classes = len(label_list)
        self.h = cfg.ldm_image_condition_h
        self.w = cfg.ldm_image_condition_w

        # State
        self.current_class_id = 1  # default to 'skin'
        self.brush_radius = 6
        self.is_generating = False
        # Painting state
        self.last_paint_pos: Optional[Tuple[int, int]] = None
        self._refresh_scheduled: bool = False
        # Undo/Redo state
        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []
        self.history_limit: int = 50

        # Mask state as class map (0..num_classes)
        self.class_map = np.zeros((self.h, self.w), dtype = np.int32)
        self.mask_img = class_map_to_rgb(self.class_map)
        self.mask_tk = ImageTk.PhotoImage(self.mask_img)

        self.generated_img: Optional[Image.Image] = None
        self.generated_tk: Optional[ImageTk.PhotoImage] = None

        # Layout frames
        self.root_frame = tk.Frame(master)
        self.root_frame.pack(fill = tk.BOTH, expand = True)

        self.left_frame = tk.Frame(self.root_frame)
        self.left_frame.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

        self.right_frame = tk.Frame(self.root_frame)
        self.right_frame.pack(side = tk.RIGHT, fill = tk.BOTH, expand = True)

        # Top-left row: action buttons
        btns_frame = tk.Frame(self.left_frame)
        btns_frame.pack(side = tk.TOP, anchor = 'nw', padx = 6, pady = 6)
        self.btn_random_prompt = tk.Button(btns_frame, text = 'Random Prompt', command = self.load_random_prompt)
        self.btn_random_prompt.pack(side = tk.LEFT, padx = 2)
        self.btn_random_mask = tk.Button(btns_frame, text = 'Random Mask', command = self.load_random_mask)
        self.btn_random_mask.pack(side = tk.LEFT, padx = 2)
        self.btn_clear_mask = tk.Button(btns_frame, text = 'Clear Mask', command = self.clear_mask)
        self.btn_clear_mask.pack(side = tk.LEFT, padx = 2)
        self.btn_refresh_mask = tk.Button(btns_frame, text = 'Refresh Mask', command = self.refresh_current_mask)
        self.btn_refresh_mask.pack(side = tk.LEFT, padx = 2)

        # Second row: prompt input
        self.prompt_var = tk.StringVar()
        self.prompt_entry = tk.Entry(self.left_frame, textvariable = self.prompt_var, width = 60)
        self.prompt_entry.pack(side = tk.TOP, anchor = 'nw', padx = 6, pady = 6)

        # Third row: cf_guidance_scale, num_inference_steps, method, eta inputs
        cf_frame = tk.Frame(self.left_frame)
        cf_frame.pack(side = tk.TOP, anchor = 'nw', padx = 6, pady = 6)
        tk.Label(cf_frame, text = 'CF Guidance Scale:').pack(side = tk.LEFT, padx = 2)
        self.cf_guidance_scale_var = tk.DoubleVar(value = 1.0)
        self.cf_guidance_scale_entry = tk.Entry(cf_frame, textvariable = self.cf_guidance_scale_var, width = 8)
        self.cf_guidance_scale_entry.pack(side = tk.LEFT, padx = 2)

        tk.Label(cf_frame, text = 'Steps:').pack(side = tk.LEFT, padx = (10, 2))
        self.num_inference_steps_var = tk.IntVar(value = 20)
        self.num_inference_steps_entry = tk.Entry(cf_frame, textvariable = self.num_inference_steps_var, width = 6)
        self.num_inference_steps_entry.pack(side = tk.LEFT, padx = 2)

        tk.Label(cf_frame, text = 'Method:').pack(side = tk.LEFT, padx = (10, 2))
        self.method_var = tk.StringVar(value = 'quadratic')
        self.method_menu = tk.OptionMenu(cf_frame, self.method_var, 'linear', 'quadratic')
        self.method_menu.pack(side = tk.LEFT, padx = 2)

        tk.Label(cf_frame, text = 'Eta:').pack(side = tk.LEFT, padx = (10, 2))
        self.eta_var = tk.DoubleVar(value = 1.0)
        self.eta_entry = tk.Entry(cf_frame, textvariable = self.eta_var, width = 6)
        self.eta_entry.pack(side = tk.LEFT, padx = 2)

        # Prepare variables for brush preview and label (preview will be created next to palette in the fourth row)
        self.brush_info_var = tk.StringVar()
        self.brush_preview_size = 125  # Enlarged to 2.5x of previous 50px
        self.brush_preview = None  # will be created in build_palette_buttons
        self.brush_label_var = tk.StringVar()
        self.update_brush_info_label()

        # Row to horizontally align mask canvas (left) and generated image (right)
        self.row_align = tk.Frame(self.root_frame)
        self.row_align.pack(side = tk.TOP, anchor = 'nw', padx = 6, pady = 6)
        self.canvas_holder = tk.Frame(self.row_align)
        self.canvas_holder.pack(side = tk.LEFT, anchor = 'nw')
        self.image_holder = tk.Frame(self.row_align)
        self.image_holder.pack(side = tk.LEFT, anchor = 'nw', padx = 6)

        # Canvas for mask (placed inside the left holder)
        self.canvas = tk.Canvas(self.canvas_holder, width = self.w, height = self.h, bg = 'black')
        self.canvas.pack(side = tk.TOP)
        self.canvas_img = self.canvas.create_image(0, 0, anchor = 'nw', image = self.mask_tk)

        # Mouse bindings
        self.canvas.bind('<Button-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_paint)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        # Right-click to pick color from the mask
        self.canvas.bind('<Button-3>', self.on_pick_color)
        # Mouse wheel for brush size (Windows/Mac)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        # Mouse wheel for many Linux setups
        self.canvas.bind('<Button-4>', self.on_mouse_wheel_linux_up)
        self.canvas.bind('<Button-5>', self.on_mouse_wheel_linux_down)
        # Undo/Redo shortcuts
        self.master.bind('<Control-z>', self.on_undo)
        self.master.bind('<Control-y>', self.on_redo)

        # Palette and brush preview placed under the mask canvas
        self.palette_frame = tk.Frame(self.canvas_holder)
        self.palette_frame.pack(side = tk.TOP, anchor = 'nw', padx = 6, pady = 6)
        self.build_palette_buttons()

        # Right panel: generated image and controls
        self.generate_btn = tk.Button(self.right_frame, text = 'Generate', command = self.on_generate, width = 20, height = 2)
        self.generate_btn.pack(side = tk.TOP, pady = 6)

        self.status_var = tk.StringVar()
        self.status_var.set('Ready')
        self.status_label = tk.Label(self.right_frame, textvariable = self.status_var)
        self.status_label.pack(side = tk.TOP, pady = 2)

        self.image_panel = tk.Label(self.image_holder)
        self.image_panel.pack(side = tk.TOP)

        # Initialize with a random mask; prompt will match the same image
        self.load_random_mask()

    def build_palette_buttons(self):
        # Container that holds brush preview (left) and palette buttons (right)
        container = tk.Frame(self.palette_frame)
        container.pack(side = tk.TOP, anchor = 'w')

        # Left: Brush size visual preview and current label
        preview_col = tk.Frame(container)
        preview_col.pack(side = tk.LEFT, anchor = 'nw', padx = 2)
        self.brush_preview = tk.Canvas(
            preview_col, width = self.brush_preview_size, height = self.brush_preview_size,
            bg = '#f0f0f0', highlightthickness = 1, highlightbackground = '#cccccc',
            )
        self.brush_preview.pack(side = tk.TOP, anchor = 'nw')
        # Label showing current brush label next to preview
        self.brush_label = tk.Label(preview_col, textvariable = self.brush_label_var)
        self.brush_label.pack(side = tk.TOP, anchor = 'nw', pady = 4)
        # Brush size textual info placed with preview (fourth row)
        self.brush_info_label = tk.Label(preview_col, textvariable = self.brush_info_var)
        self.brush_info_label.pack(side = tk.TOP, anchor = 'nw', pady = 2)

        # Right: palette buttons
        buttons_col = tk.Frame(container)
        buttons_col.pack(side = tk.LEFT, anchor = 'nw', padx = 8)

        # Background (class 0) button as an 'eraser'
        r, g, b = palette[0]
        bg_hex = '#%02x%02x%02x' % (r, g, b)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        fg_color_bg = 'white' if luminance < 128 else 'black'
        tk.Button(buttons_col, text = 'background', bg = bg_hex, fg = fg_color_bg, command = lambda: self.set_brush_class(0)).pack(side = tk.TOP, anchor = 'w', padx = 2, pady = 2)

        # Create a grid of palette buttons for semantic labels (1..18)
        grid = tk.Frame(buttons_col)
        grid.pack(side = tk.TOP, anchor = 'w')
        for i, lbl in enumerate(label_list):
            r, g, b = palette[i + 1]
            color_hex = '#%02x%02x%02x' % (r, g, b)
            # Choose white text for dark backgrounds to avoid unreadable labels (e.g., hair)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            fg_color = 'white' if luminance < 128 else 'black'
            btn = tk.Button(grid, text = lbl, bg = color_hex, fg = fg_color, command = lambda idx = i + 1: self.set_brush_class(idx))
            btn.grid(row = i // 6, column = i % 6, padx = 2, pady = 2, sticky = 'nsew')

        # Initialize preview and label now that preview widget exists
        self.update_brush_info_label()
        self.update_brush_preview()

    def set_brush_class(self, class_id: int):
        self.current_class_id = class_id
        if class_id == 0:
            self.status_var.set('Brush: background')
        else:
            self.status_var.set(f'Brush: {label_list[class_id - 1]}')
        self.update_brush_info_label()
        self.update_brush_preview()

    def on_paint(self, event):
        x, y = int(event.x), int(event.y)
        # Draw continuous stroke by interpolating between last and current positions
        if self.last_paint_pos is None:
            self._paint_circle_at(x, y)
        else:
            lx, ly = self.last_paint_pos
            self._paint_line(lx, ly, x, y)
        self.last_paint_pos = (x, y)
        # Throttle refresh to ~60 FPS
        self._schedule_refresh()

    def refresh_mask_image(self):
        self.mask_img = class_map_to_rgb(self.class_map)
        self.mask_tk = ImageTk.PhotoImage(self.mask_img)
        self.canvas.itemconfig(self.canvas_img, image = self.mask_tk)

    # -------- Painting helpers and UI updates --------
    def _paint_circle_at(self, x: int, y: int):
        r = self.brush_radius
        x0, x1 = max(0, x - r), min(self.w, x + r + 1)
        y0, y1 = max(0, y - r), min(self.h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        region = self.class_map[y0:y1, x0:x1]
        region[mask] = self.current_class_id
        self.class_map[y0:y1, x0:x1] = region

    def _paint_line(self, x0: int, y0: int, x1: int, y1: int):
        dx = x1 - x0
        dy = y1 - y0
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
        step = max(1.0, self.brush_radius * 0.5)
        steps = int(dist / step)
        for s in range(steps + 1):
            t = s / max(1, steps)
            xi = int(round(x0 + t * dx))
            yi = int(round(y0 + t * dy))
            self._paint_circle_at(xi, yi)

    def _schedule_refresh(self):
        if not self._refresh_scheduled:
            self._refresh_scheduled = True
            # ~60 FPS
            self.canvas.after(16, self._do_refresh)

    def _do_refresh(self):
        self._refresh_scheduled = False
        self.refresh_mask_image()

    def on_button_press(self, event):
        # Save state before starting a new stroke for proper Undo
        self.push_history()
        self.last_paint_pos = (int(event.x), int(event.y))
        self._paint_circle_at(self.last_paint_pos[0], self.last_paint_pos[1])
        self._schedule_refresh()

    def on_button_release(self, event):
        # Ensure final refresh after stroke ends
        self.last_paint_pos = None
        self._do_refresh()

    def update_brush_info_label(self):
        if self.current_class_id == 0:
            cls_name = 'background'
        else:
            cls_name = label_list[self.current_class_id - 1]
        self.brush_info_var.set(f'Brush: {cls_name} | size: {self.brush_radius}px')
        # Sync the preview label with current brush label
        try:
            self.brush_label_var.set(f'Current: {cls_name}')
        except Exception:
            pass

    def update_brush_preview(self):
        # Clear preview
        self.brush_preview.delete('all')
        W = H = self.brush_preview_size
        cx, cy = W // 2, H // 2
        r = int(self.brush_radius)
        # Keep circle within canvas
        r = max(1, min(r, min(cx, cy) - 4))
        # Background
        self.brush_preview.create_rectangle(0, 0, W, H, fill = '#f0f0f0', outline = '')
        # Circle color uses current class palette color for better intuition
        color_idx = max(0, min(self.current_class_id, len(palette) - 1))
        r_col, g_col, b_col = palette[color_idx]
        fill_hex = '#%02x%02x%02x' % (r_col, g_col, b_col)
        # For dark fill choose white text
        luminance = 0.299 * r_col + 0.587 * g_col + 0.114 * b_col
        text_fg = 'white' if luminance < 128 else 'black'
        # Draw circle representing brush size (diameter = 2r)
        self.brush_preview.create_oval(cx - r, cy - r, cx + r, cy + r, fill = fill_hex, outline = 'black')
        # Add size text at the center (no crosshair)
        self.brush_preview.create_text(cx, cy, text = f'{self.brush_radius}px', fill = text_fg, font = ('Arial', 10, 'bold'))

    def push_history(self):
        # Push a snapshot of current class_map to undo stack
        if len(self.undo_stack) >= self.history_limit:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.class_map.copy())
        # New action invalidates redo stack
        self.redo_stack.clear()

    def on_undo(self, event = None):
        if not self.undo_stack:
            self.status_var.set('Nothing to undo')
            return 'break'
        # Move current state to redo and restore last undo
        self.redo_stack.append(self.class_map.copy())
        self.class_map = self.undo_stack.pop()
        self.refresh_mask_image()
        self.status_var.set('Undo')
        return 'break'

    def on_redo(self, event = None):
        if not self.redo_stack:
            self.status_var.set('Nothing to redo')
            return 'break'
        # Move current to undo and restore from redo stack
        self.undo_stack.append(self.class_map.copy())
        self.class_map = self.redo_stack.pop()
        self.refresh_mask_image()
        self.status_var.set('Redo')
        return 'break'

    def on_mouse_wheel(self, event):
        # Windows/Mac: event.delta is positive when scrolled up
        delta = 1 if event.delta > 0 else -1
        self._adjust_brush_size(delta)

    def on_mouse_wheel_linux_up(self, event):
        self._adjust_brush_size(1)

    def on_mouse_wheel_linux_down(self, event):
        self._adjust_brush_size(-1)

    def _adjust_brush_size(self, delta: int):
        old = self.brush_radius
        self.brush_radius = int(np.clip(self.brush_radius + delta, 1, 128))
        if self.brush_radius != old:
            self.update_brush_info_label()
            self.update_brush_preview()

    def on_pick_color(self, event):
        """Right-click pick color (class id) from the mask at cursor position."""
        x, y = int(event.x), int(event.y)
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        try:
            picked_class = int(self.class_map[y, x])
            self.set_brush_class(picked_class)
            # Status message
            if picked_class == 0:
                self.status_var.set('Picked color: background')
            else:
                self.status_var.set(f'Picked color: {label_list[picked_class - 1]}')
        except Exception:
            pass

    def _set_right_panel_image(self, pil_img: Image.Image):
        """Resize and display a PIL image on the right panel to match mask size."""
        try:
            display_img = pil_img.resize((self.w, self.h), Image.BICUBIC)
        except Exception:
            display_img = pil_img
        self.generated_tk = ImageTk.PhotoImage(display_img)
        self.image_panel.config(image = self.generated_tk)

    def clear_mask(self):
        # Push current state to undo before clearing
        self.push_history()
        self.class_map[:, :] = 0
        self.refresh_mask_image()

    def load_random_prompt(self):
        """Pick a random caption from the current mask's corresponding image captions.
        Does not change mask or right-side image.
        """
        # Must have a current index to select prompts for current mask
        if self.current_index is None:
            self.status_var.set('No mask loaded, cannot pick prompt')
            print("[DEBUG] load_random_prompt: No current_index")
            return

        try:
            if self.dataset is not None:
                # CelebDataset stores caption file paths in `texts` and exposes `_get_captions(index)` to read them.
                captions = self.dataset._get_captions(self.current_index)
                print(f"[DEBUG] load_random_prompt: Found {len(captions) if captions else 0} captions for mask #{self.current_index}")
                if captions and len(captions) > 0:
                    selected_caption = random.choice(captions)
                    self.prompt_var.set(selected_caption)
                    self.status_var.set(f'Random prompt picked from mask #{self.current_index} captions')
                    print(f"[DEBUG] Selected caption: {selected_caption}")
                    return
        except Exception as e:
            print(f"[DEBUG] Error in load_random_prompt: {e}")
            pass

        # Fallback if dataset not available or captions missing
        self.status_var.set('No captions available for current mask')
        print("[DEBUG] Using fallback prompts")
        prompts = [
            'She is a woman with blond hair. She is wearing lipstick.',
            'A smiling man with short black hair, wearing a hat.',
            'A person with long brown hair and glasses.',
            'A portrait with red lipstick and wavy hair.',
            'A person in a blue shirt with neat hair.'
            ]
        self.prompt_var.set(prompts[np.random.randint(0, len(prompts))])
        self.status_var.set('Random prompt picked (generic fallback)')

    def load_random_mask(self):
        try:
            # Use preloaded CelebDataset if available
            dataset = self.dataset
            if dataset is None:
                dataset = CelebDataset(
                    split = 'train',
                    im_path = cfg.dataset_im_path,
                    im_size = cfg.dataset_im_size,
                    im_channels = cfg.dataset_im_channels,
                    use_latents = True,
                    latent_path = str(Path(cfg.train_task_name) / cfg.train_vqvae_latent_dir_name),
                    condition_config = cfg.condition_config,
                    )
                self.dataset = dataset

            mask_idx = np.random.randint(0, len(dataset.masks))
            self.current_index = int(mask_idx)
            mask = dataset.get_mask(mask_idx)  # CxHxW
            class_map = class_map_from_one_hot(mask)
            # Ensure expected size
            if class_map.shape != (self.h, self.w):
                class_map = np.array(Image.fromarray(class_map.astype(np.uint8), mode = 'L').resize((self.w, self.h), Image.NEAREST), dtype = np.int32)
            self.class_map = class_map.astype(np.int32)
            self.refresh_mask_image()
            # Reset history for new mask session and push initial state
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.push_history()
            self.status_var.set(f'Loaded random mask #{mask_idx}')
            self.update_brush_preview()

            # Also set the prompt to a caption corresponding to this image (if available)
            try:
                captions = dataset._get_captions(mask_idx)
                if captions and len(captions) > 0:
                    selected_caption = random.choice(captions)
                    self.prompt_var.set(selected_caption)
                    print(f"[DEBUG] Loaded caption for mask #{mask_idx}: {selected_caption}")
                else:
                    print(f"[DEBUG] No captions found for mask #{mask_idx}")
                    self.prompt_var.set('')
            except Exception as e:
                print(f"[DEBUG] Error loading caption for mask #{mask_idx}: {e}")
                self.prompt_var.set('')

            # Also load and show the corresponding original image on the right as reference
            try:
                ref_img_path = dataset.images[mask_idx]
                ref_img = Image.open(ref_img_path).convert('RGB')
                self.generated_img = ref_img
                self._set_right_panel_image(self.generated_img)
                ref_img.close()
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load random mask: {e}')

    def refresh_current_mask(self):
        """Reload the original mask for the current image index, discarding edits.
        Keeps the prompt and right-side reference image unchanged.
        """
        if self.dataset is None or self.current_index is None:
            self.status_var.set('No mask to refresh')
            return
        try:
            mask = self.dataset.get_mask(self.current_index)
            class_map = class_map_from_one_hot(mask)
            if class_map.shape != (self.h, self.w):
                class_map = np.array(
                    Image.fromarray(class_map.astype(np.uint8), mode = 'L').resize((self.w, self.h), Image.NEAREST),
                    dtype = np.int32,
                    )
            self.class_map = class_map.astype(np.int32)
            self.refresh_mask_image()
            # Reset history for refreshed mask and push initial state
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.push_history()
            self.update_brush_preview()
            self.status_var.set(f'Refreshed mask for index #{self.current_index}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to refresh mask: {e}')

    def on_generate(self):
        if self.is_generating:
            return
        self.is_generating = True
        self.generate_btn.config(state = tk.DISABLED)
        self.status_var.set('Generating... this may take a while')
        prompt_text = self.prompt_var.get().strip()
        class_map_copy = self.class_map.copy()

        def worker():
            cf_scale = self.cf_guidance_scale_var.get()
            num_steps = self.num_inference_steps_var.get()
            mask_oh = one_hot_from_class_map(class_map_copy, self.num_classes).unsqueeze(0).to(device)
            img = sample_with_mask_and_prompt(
                model = self.bundle.model,
                vae = self.bundle.vae,
                text_tokenizer = self.bundle.text_tokenizer,
                text_model = self.bundle.text_model,
                mask_oh = mask_oh,
                prompt_text = prompt_text or '',
                cf_guidance_scale = cf_scale,
                num_inference_steps = num_steps,
                method = self.method_var.get(),
                eta = float(self.eta_var.get()),
                )
            self.generated_img = img
            self.image_panel.after(0, lambda: self._set_right_panel_image(self.generated_img))
            self.status_var.set('Done')

            self.is_generating = False
            self.generate_btn.config(state = tk.NORMAL)

        threading.Thread(target = worker, daemon = True).start()


# ------------- Main entry -------------

def main(model, vqvae_ckpt):
    try:
        bundle = load_models_and_configs(model, Path(vqvae_ckpt))
    except Exception as e:
        messagebox.showerror('Initialization Error', f'Failed to load models or config: {e}')
        return

    root = tk.Tk()
    root.title('CelebHQ DDPM GUI (text + mask conditional)')
    app = MaskPainterGUI(root, bundle)
    root.mainloop()


if __name__ == '__main__':
    # ------------- Configuration variables -------------
    ldm_ckpt = 'runs_tc05/ddpm_20251024-132839/celebhq/ema_ddpm_ckpt_text_image_cond_clip.pth'
    # ldm_ckpt = 'runs_tc05_qn_train_server/ddpm_20251026-062209/LSQ/0.0800/ddpm_ckpt_text_image_cond_clip.pth'
    # ldm_ckpt = 'runs_tc05_qn_train_server/ddpm_20251026-062209/LSQ/0.0100/ddpm_ckpt_text_image_cond_clip.pth'
    # ldm_ckpt = 'runs_tc05_qn_train_server/ddpm_20251026-062209/LSQ_AnDi/0.0800/ddpm_ckpt_text_image_cond_clip.pth'
    # ldm_ckpt = 'runs_tc05_qn_train_server/ddpm_20251026-062209/LSQ_AnDi/0.0500/ddpm_ckpt_text_image_cond_clip.pth'
    vqvae_ckpt = 'model_pths/vqvae_autoencoder_ckpt_latest.pth'

    model = Unet(im_channels = cfg.autoencoder_z_channels, model_config = cfg.diffusion_model_config).to(device)
    # trainer = ProgressiveTrain(model)
    # trainer.convert_to_layers(
    #     convert_layer_type_list = reg_dict.nn_layers,
    #     tar_layer_type = 'layers_qn_lsq',
    #     noise_scale = 0.05,
    #     input_bit = 8,
    #     output_bit = 8,
    #     weight_bit = 4,
    #     )
    # trainer.add_enhance_branch_LoR(
    #     ops_factor = 0.05,
    #     )
    # trainer.add_enhance_layers(ops_factor = 0.05)
    model.load_state_dict(torch.load(ldm_ckpt))

    main(model, vqvae_ckpt)
