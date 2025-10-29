from config.ldm_scaling import build_ldm_scaling

c_factor = 2.58

# Dataset configuration
dataset_name = 'celebhq'

# Auto-detect environment: 'server' if multiple GPUs, else 'local'
import os
try:
    import torch  # type: ignore
    _gpu_count = torch.cuda.device_count()
except Exception:
    _gpu_count = 0
if os.environ.get('CFG_GPU_MSG_PRINTED', '0') != '1':
    print(f'Detected {_gpu_count} GPUs')
    os.environ['CFG_GPU_MSG_PRINTED'] = '1'

environment = 'server' if _gpu_count > 1 else 'local'

# Dataset path depends on environment
if environment == 'server':
    dataset_im_path = '/root/autodl-tmp/CelebAMask-HQ/CelebAMask-HQ'
    model_paths_ldm_ckpt_resume = '/home/SD_pytorch/runs_tc05/ddpm_20251024-132839/celebhq/ema_ddpm_ckpt_text_image_cond_clip.pth'
    train_ldm_output_root = 'runs_tc05_qn_train_server'
else:
    # Use current working directory as data path when running locally
    dataset_im_path = 'D:/datasets/CelebAMask-HQ/CelebAMask-HQ'
    model_paths_ldm_ckpt_resume = 'runs_tc05/ddpm_20251024-132839/celebhq/ema_ddpm_ckpt_text_image_cond_clip.pth'
    train_ldm_output_root = 'runs_tc05_qn_train_PC'
dataset_im_channels = 3
dataset_im_size = 256

# Diffusion configuration
diffusion_num_timesteps = 1000
diffusion_beta_start = 0.00085
diffusion_beta_end = 0.012

# Latent diffusion model configuration
ldm_scaling = build_ldm_scaling(c_factor=c_factor)
print(ldm_scaling)
ldm_down_channels = ldm_scaling['down_channels']
ldm_mid_channels = ldm_scaling['mid_channels']
ldm_down_sample = [True, True, True]
ldm_attn_down = [True, True, True]
ldm_time_emb_dim = ldm_scaling['time_emb_dim']
ldm_norm_channels = ldm_scaling['norm_channels']
ldm_num_heads = ldm_scaling['num_heads']
ldm_conv_out_channels = ldm_scaling['conv_out_channels']
ldm_num_down_layers = 2
ldm_num_mid_layers = 2
ldm_num_up_layers = 2

# Conditioning configuration
ldm_condition_types = ['text', 'image']
ldm_text_condition_text_embed_model = 'clip'
ldm_text_condition_train_text_embed_model = False
ldm_text_condition_text_embed_dim = 512
ldm_text_condition_cond_drop_prob = 0.5
ldm_image_condition_input_channels = 18
ldm_image_condition_output_channels = 3
ldm_image_condition_h = 512
ldm_image_condition_w = 512
ldm_image_condition_cond_drop_prob = 0.5

# Autoencoder configuration
autoencoder_z_channels = 4
autoencoder_codebook_size = 8192
autoencoder_down_channels = [64, 128, 256, 256]
autoencoder_mid_channels = [256, 256]
autoencoder_down_sample = [True, True, True]
autoencoder_attn_down = [False, False, False]
autoencoder_norm_channels = 32
autoencoder_num_heads = 4
autoencoder_num_down_layers = 2
autoencoder_num_mid_layers = 2
autoencoder_num_up_layers = 2

# Training configuration
train_seed = 1111
train_task_name = 'celebhq'
train_ldm_batch_size = 28
train_ldm_epochs = 500
train_num_samples = 1
train_num_grid_rows = 1
train_ldm_lr = 2e-5
train_save_latents = True
train_vqvae_latent_dir_name = 'vqvae_latents_28'
train_ldm_save_every_epochs = 30

# Model paths
model_paths_ldm_ckpt_name = 'ddpm_ckpt_text_image_cond_clip.pth'

condition_config = {
    'condition_types'       : ldm_condition_types,
    'text_condition_config' : {
        'text_embed_model'      : ldm_text_condition_text_embed_model,
        'train_text_embed_model': ldm_text_condition_train_text_embed_model,
        'text_embed_dim'        : ldm_text_condition_text_embed_dim,
        'cond_drop_prob'        : ldm_text_condition_cond_drop_prob,
        },
    'image_condition_config': {
        'image_condition_input_channels' : ldm_image_condition_input_channels,
        'image_condition_output_channels': ldm_image_condition_output_channels,
        'image_condition_h'              : ldm_image_condition_h,
        'image_condition_w'              : ldm_image_condition_w,
        'cond_drop_prob'                 : ldm_image_condition_cond_drop_prob,
        },
    }
diffusion_model_config = {
    'down_channels'    : ldm_down_channels,
    'mid_channels'     : ldm_mid_channels,
    'down_sample'      : ldm_down_sample,
    'attn_down'        : ldm_attn_down,
    'time_emb_dim'     : ldm_time_emb_dim,
    'norm_channels'    : ldm_norm_channels,
    'num_heads'        : ldm_num_heads,
    'conv_out_channels': ldm_conv_out_channels,
    'num_down_layers'  : ldm_num_down_layers,
    'num_mid_layers'   : ldm_num_mid_layers,
    'num_up_layers'    : ldm_num_up_layers,
    'condition_config' : condition_config,
    }
