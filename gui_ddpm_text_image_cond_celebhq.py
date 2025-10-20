import argparse
import threading
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

import torch
import torchvision
from torchvision.utils import make_grid

from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import get_config_value, validate_image_config, validate_text_config
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from dataset.celeb_dataset import CelebDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------- Config loader (copied from sample script for self-containment) -------------

def _as_list(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _as_str_if_path(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def load_config_from_module(module_path: str) -> Dict[str, Dict[str, Any]]:
    """Load config module and normalize keys expected by the GUI.

    This loader is tolerant to two naming styles:
    - Plain names used by earlier sampling scripts (e.g., condition_types, text_condition_*)
    - "ldm_"-prefixed names present in config/celebhq_text_image_cond.py (e.g., ldm_condition_types,
      ldm_text_condition_text_embed_model, ...)

    It also fills reasonable defaults for optional training fields that are not needed by the GUI
    (e.g., autoencoder_* training knobs). This prevents AttributeError during initialization.
    """
    module = import_module(module_path)

    def sg(name: str, default: Any = None):
        return getattr(module, name, default)

    def first_attr(names: List[str], default: Any = None):
        for n in names:
            if hasattr(module, n):
                return getattr(module, n)
        return default

    dataset_config = {
        'name'       : sg('dataset_name'),
        'im_path'    : _as_str_if_path(sg('dataset_im_path')),
        'im_channels': sg('dataset_im_channels'),
        'im_size'    : sg('dataset_im_size'),
    }

    diffusion_config = {
        'num_timesteps': sg('diffusion_num_timesteps'),
        'beta_start'   : sg('diffusion_beta_start'),
        'beta_end'     : sg('diffusion_beta_end'),
    }

    # Condition config (supports both naming styles)
    condition_types = list(first_attr(['condition_types', 'ldm_condition_types'], []))
    condition_config: Optional[Dict[str, Any]] = None
    if condition_types:
        condition_config = {'condition_types': condition_types}
        if 'text' in condition_types:
            condition_config['text_condition_config'] = {
                'text_embed_model'      : first_attr(['text_condition_text_embed_model', 'ldm_text_condition_text_embed_model']),
                'train_text_embed_model': bool(first_attr(['text_condition_train_text_embed_model', 'ldm_text_condition_train_text_embed_model'], False)),
                'text_embed_dim'        : first_attr(['text_condition_text_embed_dim', 'ldm_text_condition_text_embed_dim']),
                'cond_drop_prob'        : float(first_attr(['text_condition_cond_drop_prob', 'ldm_text_condition_cond_drop_prob'], 0.0)),
            }
        if 'image' in condition_types:
            condition_config['image_condition_config'] = {
                'image_condition_input_channels' : int(first_attr(['image_condition_input_channels', 'ldm_image_condition_input_channels'])),
                'image_condition_output_channels': int(first_attr(['image_condition_output_channels', 'ldm_image_condition_output_channels'], 3)),
                'image_condition_h'              : int(first_attr(['image_condition_h', 'ldm_image_condition_h'], sg('dataset_im_size'))),
                'image_condition_w'              : int(first_attr(['image_condition_w', 'ldm_image_condition_w'], sg('dataset_im_size'))),
                'cond_drop_prob'                 : float(first_attr(['image_condition_cond_drop_prob', 'ldm_image_condition_cond_drop_prob'], 0.0)),
            }
        if 'class' in condition_types:
            class_config: Dict[str, Any] = {}
            num_classes = first_attr(['class_condition_num_classes'], None)
            if num_classes is not None:
                class_config['num_classes'] = num_classes
            cond_drop = first_attr(['class_condition_cond_drop_prob'], None)
            if cond_drop is not None:
                class_config['cond_drop_prob'] = cond_drop
            if class_config:
                condition_config['class_condition_config'] = class_config

    ldm_config: Dict[str, Any] = {
        'down_channels'    : _as_list(sg('ldm_down_channels')),
        'mid_channels'     : _as_list(sg('ldm_mid_channels')),
        'down_sample'      : _as_list(sg('ldm_down_sample')),
        'attn_down'        : _as_list(sg('ldm_attn_down')),
        'time_emb_dim'     : sg('ldm_time_emb_dim'),
        'norm_channels'    : sg('ldm_norm_channels'),
        'num_heads'        : sg('ldm_num_heads'),
        'conv_out_channels': sg('ldm_conv_out_channels'),
        'num_down_layers'  : sg('ldm_num_down_layers'),
        'num_mid_layers'   : sg('ldm_num_mid_layers'),
        'num_up_layers'    : sg('ldm_num_up_layers'),
    }
    if condition_config is not None:
        ldm_config['condition_config'] = condition_config

    autoencoder_config = {
        'z_channels'   : sg('autoencoder_z_channels'),
        'codebook_size': sg('autoencoder_codebook_size'),
        'down_channels': _as_list(sg('autoencoder_down_channels')),
        'mid_channels' : _as_list(sg('autoencoder_mid_channels')),
        'down_sample'  : _as_list(sg('autoencoder_down_sample')),
        'attn_down'    : _as_list(sg('autoencoder_attn_down')),
        'norm_channels': sg('autoencoder_norm_channels'),
        'num_heads'    : sg('autoencoder_num_heads'),
        'num_down_layers': sg('autoencoder_num_down_layers'),
        'num_mid_layers' : sg('autoencoder_num_mid_layers'),
        'num_up_layers'  : sg('autoencoder_num_up_layers'),
    }

    # Training config: use safe defaults for missing fields
    train_config: Dict[str, Any] = {
        'seed'                     : sg('train_seed', 1111),
        'task_name'                : sg('train_task_name', 'celebhq'),
        'ldm_batch_size'           : sg('train_ldm_batch_size', 16),
        'autoencoder_batch_size'   : sg('train_autoencoder_batch_size', 16),
        'disc_start'               : sg('train_disc_start', 0),
        'disc_weight'              : sg('train_disc_weight', 0.0),
        'codebook_weight'          : sg('train_codebook_weight', 1.0),
        'commitment_beta'          : sg('train_commitment_beta', 0.25),
        'perceptual_weight'        : sg('train_perceptual_weight', 0.0),
        'kl_weight'                : sg('train_kl_weight', 0.0),
        'ldm_epochs'               : sg('train_ldm_epochs', 100),
        'autoencoder_epochs'       : sg('train_autoencoder_epochs', 0),
        'num_samples'              : sg('train_num_samples', 1),
        'num_grid_rows'            : sg('train_num_grid_rows', 1),
        'ldm_lr'                   : sg('train_ldm_lr', 1e-4),
        'autoencoder_lr'           : sg('train_autoencoder_lr', 1e-4),
        'autoencoder_acc_steps'    : sg('train_autoencoder_acc_steps', 1),
        'autoencoder_img_save_steps': sg('train_autoencoder_img_save_steps', 100),
        'save_latents'             : sg('train_save_latents', True),
        'cf_guidance_scale'        : sg('train_cf_guidance_scale', 1.0),
        'vae_latent_dir_name'      : sg('train_vae_latent_dir_name', 'vae_latents'),
        'vqvae_latent_dir_name'    : sg('train_vqvae_latent_dir_name', 'vqvae_latents'),
        'ldm_output_root'          : _as_str_if_path(sg('train_ldm_output_root', 'runs')),
        'ldm_save_every_epochs'    : sg('train_ldm_save_every_epochs', 1),
        'ldm_ckpt_name'            : _as_str_if_path(first_attr(['train_ldm_ckpt_name', 'model_paths_ldm_ckpt_name'], 'ddpm_ckpt_text_image_cond_clip.pth')),
        'vqvae_autoencoder_ckpt_name': _as_str_if_path(first_attr(['train_vqvae_autoencoder_ckpt_name', 'model_paths_vqvae_autoencoder_ckpt_name'], 'vqvae_autoencoder_ckpt.pth')),
    }

    text_encoder_ckpt = first_attr(['train_text_encoder_ckpt_name'], None)
    if text_encoder_ckpt:
        train_config['text_encoder_ckpt_name'] = _as_str_if_path(text_encoder_ckpt)

    return {
        'dataset_params'    : dataset_config,
        'diffusion_params'  : diffusion_config,
        'ldm_params'        : ldm_config,
        'autoencoder_params': autoencoder_config,
        'train_params'      : train_config,
    }


# ------------- Labels and palette -------------
label_list: List[str] = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

# Background id 0 is black; class ids 1..18 will use the following distinct colors
palette: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # background 0
    (255, 205, 148),  # skin
    (255, 255, 0),    # nose
    (0, 255, 255),    # eye_g
    (0, 128, 255),    # l_eye
    (0, 0, 255),      # r_eye
    (139, 69, 19),    # l_brow
    (160, 82, 45),    # r_brow
    (255, 105, 180),  # l_ear
    (255, 20, 147),   # r_ear
    (255, 0, 0),      # mouth
    (255, 140, 0),    # u_lip
    (178, 34, 34),    # l_lip
    (0, 0, 0),        # hair (rendered near-black)
    (128, 0, 128),    # hat
    (255, 0, 255),    # ear_r (ear ring)
    (173, 216, 230),  # neck_l
    (152, 251, 152),  # neck
    (0, 128, 0),      # cloth
]

# Ensure palette has one color per class id including background
assert len(palette) == (1 + len(label_list))


# ------------- Utility conversion functions -------------

def class_map_to_rgb(class_map: np.ndarray) -> Image.Image:
    """Convert HxW class id map (0..18) to RGB PIL image using palette."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(palette):
        rgb[class_map == cid] = color
    return Image.fromarray(rgb, mode='RGB')


def one_hot_from_class_map(class_map: np.ndarray, num_classes: int) -> torch.Tensor:
    """Create CxHxW one-hot tensor from HxW class ids (0..num_classes). Channel 0 corresponds to class id 1."""
    h, w = class_map.shape
    oh = np.zeros((num_classes, h, w), dtype=np.float32)
    for idx in range(num_classes):
        # class id for this channel is idx+1
        oh[idx] = (class_map == (idx + 1))
    return torch.from_numpy(oh)


def class_map_from_one_hot(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convert CxHxW one-hot mask to HxW class ids (0..C) where 0 is background."""
    # mask_tensor: CxHxW, values in {0,1}
    class_idx = torch.argmax(mask_tensor, dim=0)  # 0..C-1
    class_map = class_idx.to(dtype=torch.int32).cpu().numpy() + 1  # 1..C
    # background: positions where all zeros remain 0
    bg = (mask_tensor.sum(dim=0) == 0)
    class_map[bg.cpu().numpy()] = 0
    return class_map


# ------------- Diffusion sampling (adapted) -------------

def sample_with_mask_and_prompt(
    model: Unet,
    scheduler: LinearNoiseScheduler,
    train_config: Dict[str, Any],
    diffusion_model_config: Dict[str, Any],
    autoencoder_model_config: Dict[str, Any],
    diffusion_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    vae: VQVAE,
    text_tokenizer,
    text_model,
    mask_oh: torch.Tensor,
    prompt_text: str,
) -> Image.Image:
    """
    Run DDPM sampling using provided mask (1xCxHxW) and prompt text, return final decoded PIL image.
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])

    # Random latent
    xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size, im_size), device=device)

    # Text embeddings
    text_prompt = [prompt_text]
    empty_prompt = ['']
    with torch.no_grad():
        text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
        empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)

    # Prepare cond inputs
    cond_input = {'text': text_prompt_embed, 'image': mask_oh.to(device)}
    uncond_input = {'text': empty_text_embed, 'image': torch.zeros_like(mask_oh).to(device)}

    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

    # Sampling loop
    with torch.no_grad():
        for i in reversed(range(diffusion_config['num_timesteps'])):
            t = (torch.ones((xt.shape[0],), device=device) * i).long()
            noise_pred_cond = model(xt, t, cond_input)
            if cf_guidance_scale > 1:
                noise_pred_uncond = model(xt, t, uncond_input)
                noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i, device=device))

        # Decode final latent
        ims = vae.decode(xt)
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=1)
        img = torchvision.transforms.ToPILImage()(grid)
        return img


# ------------- Model bundle -------------

@dataclass
class ModelBundle:
    model: Unet
    vae: VQVAE
    scheduler: LinearNoiseScheduler
    text_tokenizer: Any
    text_model: Any
    configs: Dict[str, Dict[str, Any]]


def load_models_and_configs(
    config_module: str,
    ldm_ckpt_path: Path,
    vqvae_ckpt_path: Path,
) -> ModelBundle:
    cfg = load_config_from_module(config_module)

    diffusion_config = cfg['diffusion_params']
    dataset_config = cfg['dataset_params']
    diffusion_model_config = cfg['ldm_params']
    autoencoder_model_config = cfg['autoencoder_params']

    # Validate
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, 'Condition config required for text+image conditioning.'
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types and 'image' in condition_types, 'Both text and image conditions are required.'
    validate_text_config(condition_config)
    validate_image_config(condition_config)

    # Tokenizer and text model
    with torch.no_grad():
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']['text_embed_model'], device=device)

    # Unet
    model = Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_model_config).to(device)
    model.eval()
    if not ldm_ckpt_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {ldm_ckpt_path}')
    model.load_state_dict(torch.load(str(ldm_ckpt_path), map_location=device))

    # VQVAE
    vae = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_model_config).to(device)
    vae.eval()
    if not vqvae_ckpt_path.exists():
        raise FileNotFoundError(f'VAE checkpoint not found: {vqvae_ckpt_path}')
    vae.load_state_dict(torch.load(str(vqvae_ckpt_path), map_location=device))

    # Scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    return ModelBundle(model=model, vae=vae, scheduler=scheduler,
                       text_tokenizer=text_tokenizer, text_model=text_model, configs=cfg)


# ------------- GUI -------------

class MaskPainterGUI:
    def __init__(self, master: tk.Tk, bundle: ModelBundle):
        self.master = master
        self.bundle = bundle

        self.dataset_config = bundle.configs['dataset_params']
        self.diffusion_config = bundle.configs['diffusion_params']
        self.ldm_config = bundle.configs['ldm_params']
        self.autoencoder_config = bundle.configs['autoencoder_params']
        self.train_config = bundle.configs['train_params']

        self.num_classes = len(label_list)
        self.h = get_config_value(self.ldm_config['condition_config']['image_condition_config'], 'image_condition_h', self.dataset_config['im_size'])
        self.w = get_config_value(self.ldm_config['condition_config']['image_condition_config'], 'image_condition_w', self.dataset_config['im_size'])

        # State
        self.current_class_id = 1  # default to 'skin'
        self.brush_radius = 6
        self.is_generating = False

        # Mask state as class map (0..num_classes)
        self.class_map = np.zeros((self.h, self.w), dtype=np.int32)
        self.mask_img = class_map_to_rgb(self.class_map)
        self.mask_tk = ImageTk.PhotoImage(self.mask_img)

        self.generated_img: Optional[Image.Image] = None
        self.generated_tk: Optional[ImageTk.PhotoImage] = None

        # Layout frames
        self.root_frame = tk.Frame(master)
        self.root_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.root_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.root_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Top-left: prompt
        self.prompt_var = tk.StringVar()
        self.prompt_entry = tk.Entry(self.left_frame, textvariable=self.prompt_var, width=60)
        self.prompt_entry.pack(side=tk.TOP, anchor='nw', padx=6, pady=6)

        btns_frame = tk.Frame(self.left_frame)
        btns_frame.pack(side=tk.TOP, anchor='nw', padx=6, pady=0)

        self.btn_random_prompt = tk.Button(btns_frame, text='Random Prompt', command=self.load_random_prompt)
        self.btn_random_prompt.pack(side=tk.LEFT, padx=2)
        self.btn_random_mask = tk.Button(btns_frame, text='Random Mask', command=self.load_random_mask)
        self.btn_random_mask.pack(side=tk.LEFT, padx=2)
        self.btn_clear_mask = tk.Button(btns_frame, text='Clear Mask', command=self.clear_mask)
        self.btn_clear_mask.pack(side=tk.LEFT, padx=2)

        # Canvas for mask
        self.canvas = tk.Canvas(self.left_frame, width=self.w, height=self.h, bg='black')
        self.canvas.pack(side=tk.TOP, padx=6, pady=6)
        self.canvas_img = self.canvas.create_image(0, 0, anchor='nw', image=self.mask_tk)

        # Mouse bindings
        self.canvas.bind('<B1-Motion>', self.on_paint)
        self.canvas.bind('<Button-1>', self.on_paint)

        # Bottom-left: palette
        self.palette_frame = tk.Frame(self.left_frame)
        self.palette_frame.pack(side=tk.BOTTOM, anchor='sw', padx=6, pady=6)
        self.build_palette_buttons()

        # Right panel: generated image and controls
        self.generate_btn = tk.Button(self.right_frame, text='Generate', command=self.on_generate, width=20, height=2)
        self.generate_btn.pack(side=tk.TOP, pady=6)

        self.status_var = tk.StringVar()
        self.status_var.set('Ready')
        self.status_label = tk.Label(self.right_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.TOP, pady=2)

        self.image_panel = tk.Label(self.right_frame)
        self.image_panel.pack(side=tk.TOP, padx=6, pady=6)

        # Initialize random prompt and mask
        self.load_random_prompt()
        self.load_random_mask()

    def build_palette_buttons(self):
        # Create a grid of palette buttons
        for i, lbl in enumerate(label_list):
            color = '#%02x%02x%02x' % palette[i + 1]
            btn = tk.Button(self.palette_frame, text=lbl, bg=color, command=lambda idx=i+1: self.set_brush_class(idx))
            btn.grid(row=i // 6, column=i % 6, padx=2, pady=2, sticky='nsew')

    def set_brush_class(self, class_id: int):
        self.current_class_id = class_id
        self.status_var.set(f'Brush: {label_list[class_id-1]}')

    def on_paint(self, event):
        x, y = int(event.x), int(event.y)
        r = self.brush_radius
        x0, x1 = max(0, x - r), min(self.w, x + r + 1)
        y0, y1 = max(0, y - r), min(self.h, y + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
        region = self.class_map[y0:y1, x0:x1]
        region[mask] = self.current_class_id
        self.class_map[y0:y1, x0:x1] = region
        self.refresh_mask_image()

    def refresh_mask_image(self):
        self.mask_img = class_map_to_rgb(self.class_map)
        self.mask_tk = ImageTk.PhotoImage(self.mask_img)
        self.canvas.itemconfig(self.canvas_img, image=self.mask_tk)

    def clear_mask(self):
        self.class_map[:, :] = 0
        self.refresh_mask_image()

    def load_random_prompt(self):
        prompts = [
            'She is a woman with blond hair. She is wearing lipstick.',
            'A smiling man with short black hair, wearing a hat.',
            'A person with long brown hair and glasses.',
            'A portrait with red lipstick and wavy hair.',
            'A person in a blue shirt with neat hair.'
        ]
        self.prompt_var.set(prompts[np.random.randint(0, len(prompts))])

    def load_random_mask(self):
        try:
            # Use CelebDataset to fetch a valid mask
            cfg = self.bundle.configs
            dataset = CelebDataset(
                split='train',
                im_path=cfg['dataset_params']['im_path'],
                im_size=cfg['dataset_params']['im_size'],
                im_channels=cfg['dataset_params']['im_channels'],
                use_latents=True,
                latent_path=str(Path(cfg['train_params']['task_name']) / cfg['train_params']['vqvae_latent_dir_name']),
                condition_config=self.ldm_config['condition_config'],
            )
            mask_idx = np.random.randint(0, len(dataset.masks))
            mask = dataset.get_mask(mask_idx)  # CxHxW
            class_map = class_map_from_one_hot(mask)
            # Ensure expected size
            if class_map.shape != (self.h, self.w):
                class_map = np.array(Image.fromarray(class_map.astype(np.uint8), mode='L').resize((self.w, self.h), Image.NEAREST), dtype=np.int32)
            self.class_map = class_map.astype(np.int32)
            self.refresh_mask_image()
            self.status_var.set(f'Loaded random mask #{mask_idx}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load random mask: {e}')

    def on_generate(self):
        if self.is_generating:
            return
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED)
        self.status_var.set('Generating... this may take a while')
        prompt_text = self.prompt_var.get().strip()
        class_map_copy = self.class_map.copy()

        def worker():
            try:
                mask_oh = one_hot_from_class_map(class_map_copy, self.num_classes).unsqueeze(0).to(device)
                img = sample_with_mask_and_prompt(
                    model=self.bundle.model,
                    scheduler=self.bundle.scheduler,
                    train_config=self.train_config,
                    diffusion_model_config=self.ldm_config,
                    autoencoder_model_config=self.autoencoder_config,
                    diffusion_config=self.diffusion_config,
                    dataset_config=self.dataset_config,
                    vae=self.bundle.vae,
                    text_tokenizer=self.bundle.text_tokenizer,
                    text_model=self.bundle.text_model,
                    mask_oh=mask_oh,
                    prompt_text=prompt_text or '',
                )
                self.generated_img = img
                self.generated_tk = ImageTk.PhotoImage(self.generated_img)
                self.image_panel.after(0, lambda: self.image_panel.config(image=self.generated_tk))
                self.status_var.set('Done')
            except Exception as e:
                self.status_var.set('Failed')
                messagebox.showerror('Error', f'Generation failed: {e}')
            finally:
                self.is_generating = False
                self.generate_btn.config(state=tk.NORMAL)

        threading.Thread(target=worker, daemon=True).start()


# ------------- Main entry -------------

def main():
    parser = argparse.ArgumentParser(description='GUI for text+mask-conditional DDPM sampling on CelebHQ')
    parser.add_argument('--config', type=str, default='config.celebhq_text_image_cond', help='Config module path')
    parser.add_argument('--ldm_ckpt', type=str, default='runs/ddpm_20251019-215956/celebhq/checkpoints/epoch_005_ddpm_ckpt_text_image_cond_clip.pth', help='Path to LDM (Unet) checkpoint')
    parser.add_argument('--vqvae_ckpt', type=str, default='runs/vqvae_20251018-222220/celebhq/vqvae_autoencoder_ckpt_latest.pth', help='Path to VQVAE checkpoint')
    args = parser.parse_args()

    try:
        bundle = load_models_and_configs(args.config, Path(args.ldm_ckpt), Path(args.vqvae_ckpt))
    except Exception as e:
        messagebox.showerror('Initialization Error', f'Failed to load models or config: {e}')
        return

    root = tk.Tk()
    root.title('CelebHQ DDPM GUI (text + mask conditional)')
    app = MaskPainterGUI(root, bundle)
    root.mainloop()


if __name__ == '__main__':
    main()
