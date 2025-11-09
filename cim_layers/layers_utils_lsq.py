# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

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
    return torch.tensor(init_step_size).to(x.device)


def input_quant(module, x, isint):
    x_scale = 1.0
    if module.input_quant:
        if module.step_size_input == 1:
            module.step_size_input.data = init_step_size(x, module.input_bit)
            print(f'Initialized Step Size for Input: {module.step_size_input.data.item()}')

        x, x_scale = data_quant_lsq(
            data_float = x,
            data_bit = module.input_bit,
            step_size = module.step_size_input,
            isint = isint,
            )
    return x, x_scale


def weight_quant_noise(module, isint):
    w_q = module.weight
    w_scale = 1.0
    if module.weight_quant:
        if module.step_size_weight == 1:
            module.step_size_weight.data = init_step_size(module.weight, module.weight_bit)
            print(f'Initialized Step Size for Weight: {module.step_size_weight.data.item()}')

        w_q, w_scale = weight_quant_lsq(
            data_float = module.weight,
            data_bit = module.weight_bit,
            step_size = module.step_size_weight,
            isint = isint,
            )
    w_qn = add_noise(w_q, n_scale = module.noise_scale)
    return w_qn, w_scale


def output_quant(module, x, isint):
    x_scale = 1.0
    if module.output_quant:
        if module.step_size_output == 1:
            module.step_size_output.data = init_step_size(x, module.output_bit)
            print(f'Initialized Step Size for Output: {module.step_size_output.data.item()}')
        x, x_scale = data_quant_lsq(
            data_float = x,
            data_bit = module.output_bit,
            step_size = module.step_size_output,
            isint = isint,
            )
    return x, x_scale