import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMultiheadAttention(nn.Module):
    """
    Minimal multi-head attention module with a MultiheadAttention-compatible interface.
    Supports self- and cross-attention with batch-first inputs to replace nn.MultiheadAttention
    in existing blocks without modifying their call sites.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            key = query
        if value is None:
            value = key

        q, k, v = self._project_inputs(query, key, value)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_logits = self._apply_masks(attn_logits, attn_mask, key_padding_mask)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            attn_output.shape[0], attn_output.shape[2], self.embed_dim
        )

        attn_output = self.out_proj(attn_output)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        attn_weights_to_return: Optional[torch.Tensor] = None
        if need_weights:
            if average_attn_weights:
                attn_weights_to_return = attn_weights.mean(dim=1)
            else:
                attn_weights_to_return = attn_weights

        return attn_output, attn_weights_to_return

    def _project_inputs(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.batch_first:
            q = query
            k = key
            v = value
        else:
            q = query.transpose(0, 1)
            k = key.transpose(0, 1)
            v = value.transpose(0, 1)

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._reshape_for_heads(q)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)
        return q, k, v

    def _reshape_for_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _apply_masks(
        self,
        attn_logits: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(mask, float("-inf"))
            else:
                attn_logits = attn_logits + mask

        if key_padding_mask is not None:
            mask = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        return attn_logits
