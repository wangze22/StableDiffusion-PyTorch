# Flat Python config for VQ-VAE training on CelebHQ
# Reference: train_ddpm_cond_celebhq_new.py and config/celebhq_text_image_cond.py
try:
    import torch  # type: ignore
    _gpu_count = torch.cuda.device_count()
except Exception:
    _gpu_count = 0
print(f'Detected {_gpu_count} GPUs')

environment = 'server' if _gpu_count > 1 else 'local'

# Dataset path depends on environment
if environment == 'server':
    dataset_im_path = '/root/autodl-tmp/CelebAMask-HQ/CelebAMask-HQ'
    train_vqvae_output_root = 'runs_VQVAE_noise_server'
else:
    # Use current working directory as data path when running locally
    dataset_im_path = 'D:/datasets/CelebAMask-HQ/CelebAMask-HQ'
    train_vqvae_output_root = 'runs_VQVAE_noise_PC'

# Dataset configuration
dataset_name = 'celebhq'
dataset_im_channels = 3
dataset_im_size = 256

# Autoencoder (VQ-VAE) configuration
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
train_autoencoder_batch_size = 8
train_disc_start = -1
train_disc_weight = 5
train_codebook_weight = 1.0
train_commitment_beta = 0.2
train_perceptual_weight = 1.0
train_autoencoder_epochs = 100
train_autoencoder_lr = 1e-5
# Minimum LR for cosine annealing; if not used explicitly, script falls back to 10% of base LR

# Output and checkpoint saving
train_vqvae_save_every_epochs = 20

# Model paths
model_paths_vqvae_autoencoder_ckpt_name = 'vqvae_autoencoder_ckpt.pth'
model_paths_vqvae_discriminator_ckpt_name = 'vqvae_discriminator_ckpt.pth'
# Resume checkpoints (set to None to start fresh)
model_paths_vqvae_autoencoder_ckpt_resume = 'model_pths/vqvae_autoencoder_ckpt_latest_converged.pth'
model_paths_vqvae_discriminator_ckpt_resume = 'model_pths/vqvae_discriminator_ckpt_latest_converged.pth'


# Assembled model config dict (consumed by models.vqvae.VQVAE)
autoencoder_model_config = {
    'z_channels': autoencoder_z_channels,
    'codebook_size': autoencoder_codebook_size,
    'down_channels': autoencoder_down_channels,
    'mid_channels': autoencoder_mid_channels,
    'down_sample': autoencoder_down_sample,
    'attn_down': autoencoder_attn_down,
    'norm_channels': autoencoder_norm_channels,
    'num_heads': autoencoder_num_heads,
    'num_down_layers': autoencoder_num_down_layers,
    'num_mid_layers': autoencoder_num_mid_layers,
    'num_up_layers': autoencoder_num_up_layers,
}


# Aggregated configs for convenience
# Move dataset_config and train_config here so training scripts can import them directly

dataset_config = {
    'name': dataset_name,
    'im_path': dataset_im_path,
    'im_size': dataset_im_size,
    'im_channels': dataset_im_channels,
}

train_config = {
    'seed': train_seed,
    'autoencoder_batch_size': train_autoencoder_batch_size,
    'disc_start': train_disc_start,
    'disc_weight': train_disc_weight,
    'codebook_weight': train_codebook_weight,
    'commitment_beta': train_commitment_beta,
    'perceptual_weight': train_perceptual_weight,
    'autoencoder_epochs': train_autoencoder_epochs,
    'autoencoder_lr': train_autoencoder_lr,
    'vqvae_autoencoder_ckpt_name': model_paths_vqvae_autoencoder_ckpt_name,
    'vqvae_discriminator_ckpt_name': model_paths_vqvae_discriminator_ckpt_name,
}