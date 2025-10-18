import torch

import config.celebhq_params as cfg
from cim_weight_mapper.weight_process import map_weight_for_model
from models.unet_cond_celebhq import Unet
from cim_weight_mapper.weight_mapper import *
from cim_qn_train.progressive_qn_train import *
import cim_layers.register_dict as reg_dict
def build_condition_config() -> dict:
    """Recreate the CelebHQ conditioning configuration used by the LDM."""
    return {
        'condition_types': tuple(cfg.condition_types),
        'text_condition_config': {
            'text_embed_model': cfg.text_condition_text_embed_model,
            'train_text_embed_model': cfg.text_condition_train_text_embed_model,
            'text_embed_dim': cfg.text_condition_text_embed_dim,
            'cond_drop_prob': cfg.text_condition_cond_drop_prob,
        },
        'image_condition_config': {
            'image_condition_input_channels': cfg.image_condition_input_channels,
            'image_condition_output_channels': cfg.image_condition_output_channels,
            'image_condition_h': cfg.image_condition_h,
            'image_condition_w': cfg.image_condition_w,
            'cond_drop_prob': cfg.image_condition_cond_drop_prob,
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
        layer_details.append((name, module.__class__.__name__, param_count))
    return layer_details


def main():
    condition_config = build_condition_config()
    model_config = build_model_config(condition_config)
    model = Unet(im_channels=cfg.autoencoder_z_channels, model_config=model_config)

    layer_details = collect_layer_info(model)
    total_params = count_parameters(model)

    print(f'Total layers: {len(layer_details)}')
    print('Layer details (name | type | parameters):')
    for name, module_type, param_count in layer_details:
        print(f'- {name} | {module_type} | params: {param_count}')

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
    model_weight_mapping_info = map_weight_for_model(model,
                                                     module_for_map = reg_dict.op_layers,
                                                     draw_weight_block = True,
                                                     array_device_name = 'TC05',
                                                     array_size = [576, 2048],
                                                     weight_block_size = [576, 128])

if __name__ == '__main__':
    main()
