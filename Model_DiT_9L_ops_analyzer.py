import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

# Allow PyTorch and Intel OpenMP runtimes to coexist on Windows.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import torch.nn as nn

import Model_DiT_9L_config as cfg
from models.attention import Attention
from models.multihead_attention import CustomMultiheadAttention
from models.transformer import DIT


DEFAULT_TEXT_SEQ_LEN = 77
LATENT_DOWNSAMPLE_FACTOR = 8  # Autoencoder downsamples 2^3 times.
LATENT_PKL_DIR = 'vqvae_latents_29'
OUTPUT_JSON_PATH = 'dit_9l_ops_report.json'


@dataclass
class LayerOpStats:
    name: str
    op_type: str
    mul_ops: int = 0
    add_ops: int = 0
    attention_ops: int = 0
    matmul_ops: int = 0
    bias_ops: int = 0

    def to_dict(self) -> Dict[str, int]:
        data = asdict(self)
        data['tot_ops'] = self.mul_ops + self.add_ops + self.bias_ops + self.attention_ops
        return data


def _count_linear_ops(module: nn.Linear, input_tensor: torch.Tensor) -> Tuple[int, int, int]:
    in_features = module.in_features
    out_features = module.out_features
    if in_features == 0 or out_features == 0:
        return 0, 0, 0
    instances = input_tensor.numel() // in_features
    mul_ops = instances * in_features * out_features
    add_ops = instances * max(in_features - 1, 0) * out_features
    bias_ops = instances * out_features if module.bias is not None else 0
    return mul_ops, add_ops, bias_ops


def _count_conv2d_ops(
    module: nn.Conv2d,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
) -> Tuple[int, int, int]:
    batch = output_tensor.shape[0]
    out_channels = output_tensor.shape[1]
    out_h = output_tensor.shape[2]
    out_w = output_tensor.shape[3]
    kernel_h, kernel_w = module.kernel_size
    groups = module.groups
    in_channels = module.in_channels
    kernel_ops = (in_channels // groups) * kernel_h * kernel_w
    mul_ops = batch * out_channels * out_h * out_w * kernel_ops
    add_ops = batch * out_channels * out_h * out_w * max(kernel_ops - 1, 0)
    bias_ops = batch * out_channels * out_h * out_w if module.bias is not None else 0
    return mul_ops, add_ops, bias_ops


def _count_attention_ops(
    batch: int,
    num_heads: int,
    q_tokens: int,
    k_tokens: int,
    head_dim: int,
) -> int:
    if min(batch, num_heads, q_tokens, k_tokens, head_dim) <= 0:
        return 0
    # QK^T multiply-add
    score_mul = batch * num_heads * q_tokens * k_tokens * head_dim
    score_add = batch * num_heads * q_tokens * k_tokens * max(head_dim - 1, 0)
    scale_mul = batch * num_heads * q_tokens * k_tokens
    # (softmax is ignored because it mixes non mul/add ops such as exp)
    # AttnWeights @ V multiply-add
    attn_v_mul = batch * num_heads * q_tokens * head_dim * k_tokens
    attn_v_add = batch * num_heads * q_tokens * head_dim * max(k_tokens - 1, 0)
    return score_mul + score_add + scale_mul + attn_v_mul + attn_v_add


def _get_or_create_stats(name: str, module: nn.Module, stats: Dict[str, LayerOpStats]) -> LayerOpStats:
    if name not in stats:
        stats[name] = LayerOpStats(name=name, op_type=module.__class__.__name__)
    return stats[name]


def _ops_hook(
    module_name: str,
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    output,
    stats: Dict[str, LayerOpStats],
) -> None:
    entry = _get_or_create_stats(module_name, module, stats)
    if isinstance(module, nn.Linear):
        input_tensor = inputs[0]
        mul_ops, add_ops, bias_ops = _count_linear_ops(module, input_tensor)
        entry.mul_ops += mul_ops
        entry.add_ops += add_ops
        entry.bias_ops += bias_ops
        entry.matmul_ops += mul_ops + add_ops
    elif isinstance(module, nn.Conv2d):
        input_tensor = inputs[0]
        output_tensor = output
        mul_ops, add_ops, bias_ops = _count_conv2d_ops(module, input_tensor, output_tensor)
        entry.mul_ops += mul_ops
        entry.add_ops += add_ops
        entry.bias_ops += bias_ops
        entry.matmul_ops += mul_ops + add_ops
    elif isinstance(module, Attention):
        input_tensor = inputs[0]
        batch, tokens = input_tensor.shape[:2]
        attn_ops = _count_attention_ops(
            batch=batch,
            num_heads=module.n_heads,
            q_tokens=tokens,
            k_tokens=tokens,
            head_dim=module.head_dim,
        )
        entry.attention_ops += attn_ops
    elif isinstance(module, CustomMultiheadAttention):
        query = inputs[0]
        key = inputs[1] if len(inputs) > 1 and inputs[1] is not None else query
        value = inputs[2] if len(inputs) > 2 and inputs[2] is not None else key
        if module.batch_first:
            batch = query.shape[0]
            q_tokens = query.shape[1]
            k_tokens = key.shape[1]
        else:
            q_tokens = query.shape[0]
            k_tokens = key.shape[0]
            batch = query.shape[1]
        attn_ops = _count_attention_ops(
            batch=batch,
            num_heads=module.num_heads,
            q_tokens=q_tokens,
            k_tokens=k_tokens,
            head_dim=module.head_dim,
        )
        entry.attention_ops += attn_ops


def _register_hooks(model: nn.Module, stats: Dict[str, LayerOpStats]):
    handles = []
    for name, module in model.named_modules():
        if not name:
            continue

        def hook_fn(mod, inp, out, module_name=name):
            _ops_hook(module_name, mod, inp, out, stats)

        handles.append(module.register_forward_hook(hook_fn))
    return handles


def _load_sample_latent(device: torch.device) -> Optional[torch.Tensor]:
    if not os.path.isdir(LATENT_PKL_DIR):
        return None
    pkl_files = sorted(
        [
            os.path.join(LATENT_PKL_DIR, f)
            for f in os.listdir(LATENT_PKL_DIR)
            if f.endswith('.pkl')
        ]
    )
    for path in pkl_files:
        try:
            with open(path, 'rb') as fp:
                payload = torch.load(fp, map_location='cpu')
        except Exception:
            try:
                import pickle

                with open(path, 'rb') as fp:  # fallback to pickle module
                    payload = pickle.load(fp)
            except Exception:
                continue
        if isinstance(payload, dict) and payload:
            latent = next(iter(payload.values()))
        elif torch.is_tensor(payload):
            latent = payload
        else:
            continue
        if not torch.is_tensor(latent):
            continue
        return latent.to(device)
    return None


def _prepare_dummy_inputs(
    batch_size: int,
    latent_hw: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    latent_sample = _load_sample_latent(device)
    if latent_sample is None:
        height = width = latent_hw
        latent = torch.randn(batch_size, cfg.autoencoder_z_channels, height, width, device=device)
    else:
        latent = latent_sample
        height = latent.shape[-2]
        width = latent.shape[-1]
        if latent.shape[0] != batch_size:
            latent = latent.repeat(batch_size, 1, 1, 1)
    timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)

    cond_input: Dict[str, torch.Tensor] = {}
    if 'text' in cfg.ldm_condition_types:
        cond_input['text'] = torch.zeros(
            batch_size,
            DEFAULT_TEXT_SEQ_LEN,
            cfg.ldm_text_condition_text_embed_dim,
            device=device,
        )
    if 'image' in cfg.ldm_condition_types:
        cond_input['image'] = torch.zeros(
            batch_size,
            cfg.ldm_image_condition_input_channels,
            height,
            width,
            device=device,
        )
    return latent, timesteps, cond_input or None


def analyze_ops(model: DIT, json_output_path: str) -> List[Dict[str, int]]:
    device = next(model.parameters()).device
    latent_hw = cfg.dataset_im_size // LATENT_DOWNSAMPLE_FACTOR
    latent, timesteps, cond_input = _prepare_dummy_inputs(batch_size=1, latent_hw=latent_hw, device=device)

    stats: Dict[str, LayerOpStats] = {}
    handles = _register_hooks(model, stats)
    model.eval()
    with torch.no_grad():
        model(latent, timesteps, cond_input)
    for handle in handles:
        handle.remove()

    serialized = [stats[name].to_dict() for name in stats]
    with open(json_output_path, 'w', encoding='utf-8') as fp:
        json.dump(serialized, fp, indent=2)
    return serialized


def compute_attention_share_from_json(json_path: str) -> float:
    with open(json_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    total_ops = sum(item.get('tot_ops', 0) for item in data)
    total_attention = sum(item.get('attention_ops', 0) for item in data)
    if total_ops == 0:
        return 0.0
    return total_attention / total_ops


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DIT(
        im_channels=cfg.autoencoder_z_channels,
        model_config=cfg.dit_model_config,
    ).to(device)

    layer_stats = analyze_ops(model, OUTPUT_JSON_PATH)
    total_ops = sum(item['tot_ops'] for item in layer_stats)
    attention_ops = sum(item['attention_ops'] for item in layer_stats)
    print(f'OPS analysis written to {OUTPUT_JSON_PATH}')
    print(f'Total ops: {total_ops:,}')
    print(f'Attention ops: {attention_ops:,} ({attention_ops / total_ops:.2%} of total)' if total_ops else 'Attention ops: 0')

    attention_share = compute_attention_share_from_json(OUTPUT_JSON_PATH)
    print(f'Attention share recomputed from JSON: {attention_share:.2%}')


if __name__ == '__main__':
    main()
