c_factor = 4

# Dataset configuration
dataset_name = 'celebhq'
dataset_im_path = 'D:/datasets/CelebAMask-HQ/CelebAMask-HQ'
dataset_im_channels = 3
dataset_im_size = 256

# Diffusion configuration
diffusion_num_timesteps = 1000
diffusion_beta_start = 0.00085
diffusion_beta_end = 0.012

# Latent diffusion model configuration
ldm_down_channels = [256 // c_factor, 384 // c_factor, 512 // c_factor, 768 // c_factor]
ldm_mid_channels = [768 // c_factor, 512 // c_factor]
ldm_down_sample = [True, True, True]
ldm_attn_down = [True, True, True]
ldm_time_emb_dim = 512
ldm_norm_channels = 32 // c_factor
ldm_num_heads = 16
ldm_conv_out_channels = 128 // c_factor
ldm_num_down_layers = 2
ldm_num_mid_layers = 2
ldm_num_up_layers = 2

# Conditioning configuration
ldm_condition_types = ['text', 'image']
ldm_text_condition_text_embed_model = 'clip'
ldm_text_condition_train_text_embed_model = False
ldm_text_condition_text_embed_dim = 512
ldm_text_condition_cond_drop_prob = 0.1
ldm_image_condition_input_channels = 18
ldm_image_condition_output_channels = 3
ldm_image_condition_h = 512
ldm_image_condition_w = 512
ldm_image_condition_cond_drop_prob = 0.1

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
train_ldm_batch_size = 32
train_ldm_epochs = 100
train_ldm_output_root = 'runs'
train_num_samples = 1
train_num_grid_rows = 1
train_ldm_lr = 1e-4
train_save_latents = True
train_cf_guidance_scale = 1.0
train_vae_latent_dir_name = 'vae_latents'
train_vqvae_latent_dir_name = 'vqvae_latents'
train_ldm_save_every_epochs = 1

# Model paths
model_paths_ldm_ckpt_name = 'ddpm_ckpt_text_image_cond_clip.pth'
model_paths_ldm_ckpt_resume = 'model_pths/ddpm_ckpt_text_image_cond_clip_latest.pth'
model_paths_vqvae_autoencoder_ckpt_name = 'vqvae_autoencoder_ckpt.pth'
model_paths_vae_autoencoder_ckpt_name = 'vae_autoencoder_ckpt.pth'
model_paths_vqvae_discriminator_ckpt_name = 'vqvae_discriminator_ckpt.pth'
model_paths_vae_discriminator_ckpt_name = 'vae_discriminator_ckpt.pth'
