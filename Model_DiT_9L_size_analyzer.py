import os

# Allow PyTorch and Intel OpenMP runtimes to coexist on Windows.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import Model_DiT_9L_config as cfg
from models.transformer import DIT
from cim_weight_mapper.weight_process import map_weight_for_model
import cim_layers.register_dict as reg_dict


def build_condition_config() -> dict:
    """Recreate the CelebHQ conditioning configuration used by the DiT model."""
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

    # Create DiT model
    model = DIT(
        im_channels=cfg.autoencoder_z_channels,
        model_config=cfg.dit_model_config,
    )

    layer_details = collect_layer_info(model)
    total_params = count_parameters(model)

    print(f'Total layers: {len(layer_details)}')
    print('Layer details (name | type | parameters):')
    for name, module_type, param_count, weight_shape in layer_details:
        # 添加权重形状信息到输出中
        shape_str = f" | shape: {list(weight_shape)}" if weight_shape is not None else ""
        print(f'- {name} | {module_type} | params: {param_count/1e6:.2f} M{shape_str}')

    print(f'\nTotal parameters: {total_params/1e6:.2f} M')
    if total_params < 576*2048*16:
        # Analyze weight mapping
        array_size = [576, 2048]
        print('\nMapping weights to CIM arrays...')
        try:
            model_weight_mapping_info = map_weight_for_model(
                model,
                array_size=array_size,
                weight_block_size=array_size,
                module_for_map=reg_dict.op_layers,
                draw_weight_block=True,
                array_device_name='TC05-DiT-9L'
            )
            print('Weight mapping completed successfully.')
            print('Check the "Array_Mapping_Info_(TC05-DiT-9L)" directory for visualization.')
        except Exception as e:
            print(f'Error during weight mapping: {e}')


if __name__ == '__main__':
    main()