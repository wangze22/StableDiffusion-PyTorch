import os

# Allow PyTorch and Intel OpenMP runtimes to coexist on Windows.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch

import config.celebhq_text_image_cond_tc05 as cfg
from cim_weight_mapper.weight_process import map_weight_for_model
from models.unet_cond_base_relu import Unet
from cim_weight_mapper.weight_mapper import *
from cim_qn_train.progressive_qn_train import *
import cim_layers.register_dict as reg_dict

def build_condition_config() -> dict:
    """Recreate the CelebHQ conditioning configuration used by the LDM (updated for celebhq_text_image_cond.py)."""
    return {
        'condition_types': tuple(cfg.ldm_condition_types),
        'text_condition_config': {
            'text_embed_model': cfg.ldm_text_condition_text_embed_model,
            'train_text_embed_model': cfg.ldm_text_condition_train_text_embed_model,
            'text_embed_dim': cfg.ldm_text_condition_text_embed_dim,
            'cond_drop_prob': cfg.ldm_text_condition_cond_drop_prob,
        },
        'image_condition_config': {
            'image_condition_input_channels': cfg.ldm_image_condition_input_channels,
            'image_condition_output_channels': cfg.ldm_image_condition_output_channels,
            'image_condition_h': cfg.ldm_image_condition_h,
            'image_condition_w': cfg.ldm_image_condition_w,
            'cond_drop_prob': cfg.ldm_image_condition_cond_drop_prob,
        },
    }


def build_model_config(condition_config: dict) -> dict:
    """Prepare the Unet model configuration using CelebHQ parameters."""
    model_config = {
        'down_channels': list(cfg.ldm_down_channels),
        'mid_channels': list(cfg.ldm_mid_channels),
        'down_sample': list(cfg.ldm_down_sample),
        'attn_down': list(cfg.ldm_attn_down),
        'time_emb_dim': cfg.ldm_time_emb_dim,
        'norm_channels': cfg.ldm_norm_channels,
        'num_heads': cfg.ldm_num_heads,
        'conv_out_channels': cfg.ldm_conv_out_channels,
        'num_down_layers': cfg.ldm_num_down_layers,
        'num_mid_layers': cfg.ldm_num_mid_layers,
        'num_up_layers': cfg.ldm_num_up_layers,
        'condition_config': condition_config,
    }
    return model_config


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def collect_layer_info(model: torch.nn.Module):
    """Gather per-layer metadata without double-counting nested parameters."""
    layer_details = []
    for name, module in model.named_modules():
        if name == '':
            # Skip the root module to avoid redundant reporting.
            continue
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        # 获取模块的权重形状信息（如果存在）
        weight_shape = getattr(module, 'weight', None)
        layer_details.append((name, module.__class__.__name__, param_count, 
                            weight_shape.shape if weight_shape is not None else None))
    return layer_details


def main():
    condition_config = build_condition_config()
    model_config = build_model_config(condition_config)
    model = Unet(im_channels=cfg.autoencoder_z_channels, model_config=model_config)

    layer_details = collect_layer_info(model)
    total_params = count_parameters(model)

    print(f'Total layers: {len(layer_details)}')
    print('Layer details (name | type | parameters):')
    for name, module_type, param_count, weight_shape in layer_details:
        # 添加权重形状信息到输出中
        shape_str = f" | shape: {list(weight_shape)}" if weight_shape is not None else ""
        print(f'- {name} | {module_type} | params: {param_count}{shape_str}')

    print(f'Total parameters: {total_params/1e6:.2f} M')

    # converter = ProgressiveTrain(model, device = 'cuda')
    # converter.convert_to_layers(
    #     convert_layer_type_list = reg_dict.op_layers,
    #     tar_layer_type = 'layers_qn_lsq_adda_cim',
    #     noise_scale = 0,
    #     input_bit = 8,
    #     output_bit = 8,
    #     weight_bit = 4,
    #     adc_bit = 8,
    #     dac_bit = 5,
    #     adc_gain_1_scale = 1 / 63,
    #     adc_gain_range = [1, 255],
    #     adc_adjust_mode = 'gain'
    #     )
    array_size = [576,2048]
    model_weight_mapping_info = map_weight_for_model(model,
                                                     module_for_map = reg_dict.op_layers,
                                                     draw_weight_block = True,
                                                     array_device_name = 'TC05',
                                                     array_size = array_size,
                                                     weight_block_size = array_size)

if __name__ == '__main__':
    main()
