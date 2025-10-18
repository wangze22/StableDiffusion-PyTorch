from cim_layers.quant_noise_utils import *


def update_step_size(module, weight_bit_old, input_bit_old, output_bit_old):
    if module.weight_bit != weight_bit_old:
        weight_step_size_factor = 2 ** (module.weight_bit - weight_bit_old)
        step_size_weight_old = module.step_size_weight.data.item()
        module.step_size_weight.data /= weight_step_size_factor
        print(f'step_size_weight changed: {step_size_weight_old} -> {module.step_size_weight.data}')

    if module.input_bit != input_bit_old:
        input_step_size_factor = 2 ** (module.input_bit - input_bit_old)
        step_size_input_old = module.step_size_input.data.item()
        module.step_size_input.data /= input_step_size_factor
        print(f'step_size_input changed: {step_size_input_old} -> {module.step_size_input.data}')

    if module.output_bit != output_bit_old:
        output_step_size_factor = 2 ** (module.output_bit - output_bit_old)
        step_size_output_old = module.step_size_output.data.item()
        module.step_size_output.data /= output_step_size_factor
        print(f'step_size_output changed: {step_size_output_old} -> {module.step_size_output.data}')


def init_step_size(x, data_bit):
    _, scale = data_quant(x, data_bit = data_bit, isint = True)
    init_step_size = 1 / scale
    return init_step_size
