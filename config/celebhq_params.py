from pathlib import Path

# ===============
# Dataset Params
# ===============
dataset_name = 'celebhq'
dataset_im_path = Path('D:/datasets/CelebAMask-HQ/CelebAMask-HQ')
dataset_im_channels = 3
dataset_im_size = 256

# ===============
# Diffusion Params
# ===============
diffusion_num_timesteps = 1000
diffusion_beta_start = 0.00085
diffusion_beta_end = 0.012

# ===============
# Condition Params
# ===============
condition_types = ('text', 'image')

# ---- TextConditionParams ----
text_condition_text_embed_model = 'clip'
text_condition_train_text_embed_model = False
text_condition_text_embed_dim = 512
text_condition_cond_drop_prob = 0.1

# ---- ImageConditionParams ----
image_condition_input_channels = 18
image_condition_output_channels = 3
image_condition_h = 512
image_condition_w = 512
image_condition_cond_drop_prob = 0.1

# ===============
# LDM Params
# ===============
channel_scale_factor = 4
head_scale_factor = 1

# scale down channel counts by the configured factor
ldm_down_channels = tuple(channel // channel_scale_factor for channel in (256, 384, 512, 768))
ldm_mid_channels = tuple(channel // channel_scale_factor for channel in (768, 512))
ldm_down_sample = (True, True, True)
ldm_attn_down = (True, True, True)
ldm_time_emb_dim = 512
ldm_norm_channels = 32 // channel_scale_factor
ldm_num_heads = 16 // head_scale_factor
ldm_conv_out_channels = 128 // channel_scale_factor
ldm_num_down_layers = 2
ldm_num_mid_layers = 2
ldm_num_up_layers = 2

# ===============
# Autoencoder Params
# ===============
autoencoder_z_channels = 4
autoencoder_codebook_size = 8192
autoencoder_down_channels = (64, 128, 256, 256)
autoencoder_mid_channels = (256, 256)
autoencoder_down_sample = (True, True, True)
autoencoder_attn_down = (False, False, False)
autoencoder_norm_channels = 32
autoencoder_num_heads = 4
autoencoder_num_down_layers = 2
autoencoder_num_mid_layers = 2
autoencoder_num_up_layers = 2

# ===============
# Train Params
# ===============
train_seed = 1111
train_task_name = 'celebhq'
train_ldm_batch_size = 32
train_autoencoder_batch_size = 4
train_disc_start = 15000
train_disc_weight = 5.0
train_codebook_weight = 1.0
train_commitment_beta = 0.2
train_perceptual_weight = 1.0
train_kl_weight = 0.000005
train_ldm_epochs = 100
train_autoencoder_epochs = 20
train_num_samples = 1
train_num_grid_rows = 1
train_ldm_lr = 1e-4
train_autoencoder_lr = 0.00001
train_autoencoder_acc_steps = 4
train_autoencoder_img_save_steps = 64
train_save_latents = True
train_cf_guidance_scale = 1.0
train_vae_latent_dir_name = 'vae_latents'
train_vqvae_latent_dir_name = 'vqvae_latents'
train_ldm_output_root = Path('runs')
train_ldm_save_every_epochs = 5
train_ldm_ckpt_name = 'ddpm_ckpt_text_image_cond_clip.pth'
train_text_encoder_ckpt_name = 'text_encoder_ckpt.pth'
train_vqvae_autoencoder_ckpt_name = Path('runs') / 'vqvae_20251018-222220' / 'celebhq' / 'vqvae_autoencoder_ckpt_latest.pth'
