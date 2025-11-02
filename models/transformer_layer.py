import torch.nn as nn
from models.attention import Attention
from models.multihead_attention import CustomMultiheadAttention


class TransformerLayer(nn.Module):
    r"""
    Transformer block which is just doing the following based on VIT
        1. LayerNorm followed by Attention
        2. LayerNorm followed by Feed forward Block
        Both these also have residuals added to them

        For DiT we additionally have
        1. Layernorm mlp to predict layernorm affine parameters from
        2. Same Layernorm mlp to also predict scale parameters for outputs
            of both mlp/attention prior to residual connection.
    """

    def __init__(self, config, *, cross_attn = False, context_dim = None):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.cross_attn = cross_attn
        self.context_dim = context_dim

        ff_hidden_dim = 4 * self.hidden_size

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.hidden_size, elementwise_affine = False, eps = 1E-6)

        self.attn_block = Attention(config)

        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine = False, eps = 1E-6)

        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, self.hidden_size),
            )

        if self.cross_attn:
            assert self.context_dim is not None, "Context dimension must be provided for cross attention"
            self.cross_attn_norm = nn.LayerNorm(self.hidden_size, elementwise_affine = False, eps = 1E-6)
            self.cross_attn_block = CustomMultiheadAttention(
                self.hidden_size,
                config['num_heads'],
                batch_first = True,
                )
            self.context_proj = nn.Linear(self.context_dim, self.hidden_size)

        # Scale Shift Parameter predictions for this layer
        # 1. Scale and shift parameters for layernorm of attention (2 * hidden_size)
        # 2. Scale and shift parameters for layernorm of mlp (2 * hidden_size)
        # 3. Scale for output of attention prior to residual connection (hidden_size)
        # 4. Scale for output of mlp prior to residual connection (hidden_size)
        # Total 6 * hidden_size
        # When cross-attention is enabled, we add another (2 * hidden_size) + hidden_size prediction
        # for the cross-attention norm and residual scale.
        adaptive_out_dim = 6 * self.hidden_size
        self.adaptive_norm_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, adaptive_out_dim, bias = True),
            )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        if self.cross_attn:
            nn.init.xavier_uniform_(self.context_proj.weight)
            nn.init.constant_(self.context_proj.bias, 0)

    def forward(self, x, condition, context = None):
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim = 1)
        (
            pre_attn_shift, pre_attn_scale, post_attn_scale,
            pre_mlp_shift, pre_mlp_scale, post_mlp_scale
            ) = scale_shift_params

        out = x
        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1))
                            + pre_attn_shift.unsqueeze(1))
        out = out + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_output)

        if self.cross_attn and context is not None:
            context_tokens = self.context_proj(context)
            cross_norm_output = self.cross_attn_norm(out)
            cross_attn_out, _ = self.cross_attn_block(
                cross_norm_output,
                context_tokens,
                context_tokens,
                need_weights = False,
                )
            out = out + cross_attn_out

        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
                           pre_mlp_shift.unsqueeze(1))
        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)
        return out
