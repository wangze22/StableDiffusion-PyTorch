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
    w_qn = module.weight
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

#
# import math
# import torch
#
#
# def _maybe_mark_inited_from_loaded(step_param, flag_name: str, module):
#     """
#     若 step_param 已从 checkpoint 加载为 “非 1.0”，但 flag 未设置，则将 flag 置 True，避免重新初始化。
#     仅在首次检查时做一次 CPU 同步用于判断（detach().cpu()），之后不再同步。
#     """
#     if getattr(module, flag_name, None):
#         return  # 已经标记过
#
#     # 仅在首次进入时检查一次
#     if torch.is_tensor(step_param) and step_param.numel() == 1:
#         v = float(step_param.detach().cpu())  # 一次性同步
#         if not math.isclose(v, 1.0, rel_tol = 0.0, abs_tol = 1e-12):
#             setattr(module, flag_name, True)  # 认为是从权重中加载的有效步长
#             return
#     # 否则保持 flag 缺省/False，交由后续初始化逻辑处理
#
#
# def input_quant(module, x, isint):
#     x_scale = 1.0
#     if module.input_quant:
#         # 1) 若 checkpoint 已把 step_size_input 设成非 1，但 flag 未置位，则自动置位
#         _maybe_mark_inited_from_loaded(module.step_size_input, "_inited_step_size_input", module)
#
#         # 2) 如仍未初始化，则按需要初始化一次（使用当前 x）
#         if not getattr(module, "_inited_step_size_input", False):
#             with torch.no_grad():
#                 module.step_size_input.data = init_step_size(x, module.input_bit)
#             setattr(module, "_inited_step_size_input", True)
#
#         # 3) 量化
#         x, x_scale = data_quant_lsq(
#             data_float = x,
#             data_bit = module.input_bit,
#             step_size = module.step_size_input,
#             isint = isint,
#             )
#     return x, x_scale
#
#
# def weight_quant_noise(module, isint):
#     w_qn = module.weight
#     w_scale = 1.0
#     if module.weight_quant:
#         # 1) 若 checkpoint 已把 step_size_weight 设成非 1，但 flag 未置位，则自动置位
#         _maybe_mark_inited_from_loaded(module.step_size_weight, "_inited_step_size_weight", module)
#
#         # 2) 如仍未初始化，则按需要初始化一次（使用当前 weight）
#         if not getattr(module, "_inited_step_size_weight", False):
#             with torch.no_grad():
#                 module.step_size_weight.data = init_step_size(module.weight, module.weight_bit)
#             setattr(module, "_inited_step_size_weight", True)
#
#         # 3) 权重量化 + 噪声
#         w_q, w_scale = weight_quant_lsq(
#             data_float = module.weight,
#             data_bit = module.weight_bit,
#             step_size = module.step_size_weight,
#             isint = isint,
#             )
#         w_qn = add_noise(w_q, n_scale = module.noise_scale)
#     return w_qn, w_scale
#
#
# def output_quant(module, x, isint):
#     x_scale = 1.0
#     if module.output_quant:
#         # 1) 若 checkpoint 已把 step_size_output 设成非 1，但 flag 未置位，则自动置位
#         _maybe_mark_inited_from_loaded(module.step_size_output, "_inited_step_size_output", module)
#
#         # 2) 如仍未初始化，则按需要初始化一次（使用当前 x）
#         if not getattr(module, "_inited_step_size_output", False):
#             with torch.no_grad():
#                 module.step_size_output.data = init_step_size(x, module.output_bit)
#             setattr(module, "_inited_step_size_output", True)
#
#         # 3) 量化
#         x, x_scale = data_quant_lsq(
#             data_float = x,
#             data_bit = module.output_bit,
#             step_size = module.step_size_output,
#             isint = isint,
#             )
#     return x, x_scale
