import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from models.patch_embed import PatchEmbedding
from models.transformer_layer import TransformerLayer
from utils.config_utils import (
    get_config_value,
    validate_class_config,
    validate_class_conditional_input,
    validate_image_config,
    validate_image_conditional_input,
    validate_text_config,
)


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DIT(nn.Module):
    def __init__(self, im_channels, model_config, image_size=None):
        super().__init__()

        self.image_height = image_size
        self.image_width = image_size
        self.im_channels = im_channels

        self.hidden_size = model_config['hidden_size']
        self.patch_height = model_config['patch_size']
        self.patch_width = model_config['patch_size']
        self.timestep_emb_dim = model_config['timestep_emb_dim']
        self.num_layers = model_config['num_layers']
        self.num_heads = model_config['num_heads']
        self.head_dim = model_config['head_dim']

        ######## Class, Mask and Text Conditioning Config #####
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.text_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
            condition_types = self.condition_config['condition_types']
            if 'class' in condition_types:
                validate_class_config(self.condition_config)
                self.class_cond = True
                self.num_classes = self.condition_config['class_condition_config']['num_classes']
            if 'text' in condition_types:
                validate_text_config(self.condition_config)
                self.text_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                validate_image_config(self.condition_config)
                self.image_cond = True
                image_cfg = self.condition_config['image_condition_config']
                self.im_cond_input_ch = image_cfg['image_condition_input_channels']
                self.im_cond_output_ch = image_cfg['image_condition_output_channels']
        if self.class_cond:
            self.class_emb = nn.Embedding(self.num_classes, self.timestep_emb_dim)

        if self.image_cond:
            # Map the conditioning image to a set of channels we can concat with the latent
            self.cond_conv_in = nn.Conv2d(
                in_channels=self.im_cond_input_ch,
                out_channels=self.im_cond_output_ch,
                kernel_size=1,
                bias=False,
            )
            patch_embed_in_channels = im_channels + self.im_cond_output_ch
        else:
            patch_embed_in_channels = im_channels
        self.cond = self.text_cond or self.image_cond or self.class_cond
        ###################################

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(image_height=self.image_height or 0,
                                                image_width=self.image_width or 0,
                                                im_channels=patch_embed_in_channels,
                                                patch_height=self.patch_height,
                                                patch_width=self.patch_width,
                                                hidden_size=self.hidden_size)

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # All Transformer Layers
        layer_config = {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
        }
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                layer_config,
                cross_attn=self.text_cond,
                context_dim=self.text_embed_dim if self.text_cond else None,
            ) for _ in range(self.num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        # Final Linear Layer
        self.proj_out = nn.Linear(self.hidden_size,
                                  self.patch_height * self.patch_width * self.im_channels)

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t, cond_input=None):
        if self.cond:
            assert cond_input is not None, \
                "Model initialized with conditioning so cond_input cannot be None"

        patch_source = x
        if self.image_cond:
            validate_image_conditional_input(cond_input, x)
            im_cond = cond_input['image'].to(device=x.device, dtype=x.dtype)
            im_cond = F.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            patch_source = torch.cat([patch_source, im_cond], dim=1)

        # Patchify
        out = self.patch_embed_layer(patch_source)

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_tensor = torch.as_tensor(t, device=x.device)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.unsqueeze(0)
        t_emb = get_time_embedding(t_tensor.long(), self.timestep_emb_dim)

        ######## Class Conditioning ########
        if self.class_cond:
            validate_class_conditional_input(cond_input, x, self.num_classes)
            class_tokens = cond_input['class'].to(device=t_emb.device, dtype=self.class_emb.weight.dtype)
            class_embed = einsum(class_tokens, self.class_emb.weight, 'b n, n d -> b d')
            t_emb += class_embed
        ####################################

        # (Batch, timestep_emb_dim) -> (Batch, hidden_size)
        t_emb = self.t_proj(t_emb)

        context_hidden_states = None
        if self.text_cond:
            assert 'text' in cond_input, \
                "Model initialized with text conditioning but cond_input has no text information"
            context_hidden_states = cond_input['text'].to(device=out.device, dtype=out.dtype)

        # Go through the transformer layers
        for layer in self.transformer_layers:
            out = layer(out, t_emb, context_hidden_states)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(t_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,hidden_size) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)
        _, _, height, width = x.shape
        nh = height // self.patch_height
        nw = width // self.patch_width
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=nw,
                        nh=nh)
        return out
