# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

from cim_layers.quant_noise_utils import *
import math


def init_adc_gain(module, out_, adc_adjust_mode = 'gain'):
    if out_.abs().max() != 0:
        adc_scale_ideal = module.adc_range / out_.abs().max()
        adc_gain_ideal = adc_scale_ideal / module.adc_gain_1_scale
        adc_gain_ideal = torch.clamp(adc_gain_ideal, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)

        module.adc_gain.data.copy_(adc_gain_ideal.to(module.adc_gain.device))
        module.adc_scale = get_adc_scale(module, module.adc_gain, adc_adjust_mode)
        print(f'Initialized Adc Gain: {module.adc_gain.data}')


def init_adc_gain_(module, out_):
    if out_.abs().max() != 0:
        adc_scale_ideal = module.adc_range / out_.abs().max()
        adc_gain_ideal = adc_scale_ideal / module.adc_gain_1_scale
        adc_gain_ideal = torch.clamp(adc_gain_ideal, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)
        print(f'Initialized Adc Gain: {adc_gain_ideal}')
    else:
        return None
    return adc_gain_ideal


# def get_adc_scale(module, adc_gain):
#     adc_gain = torch.clamp(adc_gain, min = module.adc_gain_min, max = module.adc_gain_max)
#     adc_gain = round_pass(adc_gain)
#     adc_scale = adc_gain * module.adc_gain_1_scale
#     return adc_scale

def get_adc_scale(module, adc_gain, mode = 'gain'):
    # adc_gain = torch.clamp(adc_gain, min = module.adc_gain_min, max = module.adc_gain_max)
    adc_gain = clamp_pass(adc_gain, min = module.adc_gain_min, max = module.adc_gain_max)
    if mode == 'gain':
        adc_gain = round_pass(adc_gain)
    else:
        adc_range = round_pass(1 / adc_gain)
        adc_gain = 1 / adc_range
    adc_scale = adc_gain * module.adc_gain_1_scale
    return adc_scale


def update_adc_gain(module, adc_bit_old, dac_bit_old, weight_bit_old):
    adc_gain_old = module.adc_gain.data.item()
    adc_gain_new = adc_gain_old
    if adc_bit_old != module.adc_bit:
        adc_range_factor = 2 ** (module.adc_bit - adc_bit_old)
        adc_gain_new *= adc_range_factor

    if dac_bit_old != module.dac_bit:
        dac_range_factor = 2 ** (module.dac_bit - dac_bit_old)
        adc_gain_new /= dac_range_factor

    if weight_bit_old != module.weight_bit:
        weight_range_factor = 2 ** (module.weight_bit - weight_bit_old)
        adc_gain_new /= weight_range_factor

    adc_gain_new = torch.tensor(adc_gain_new, device = module.adc_gain.device)
    adc_gain_new = torch.clamp(adc_gain_new, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)

    module.adc_gain.data = adc_gain_new

    if adc_gain_old != module.adc_gain.data.item():
        print(f'adc_gain changed: {adc_gain_old} -> {adc_gain_new}')


def update_adc_gain_multi(module, adc_bit_old, dac_bit_old, weight_bit_old):
    for key, adc_gain_old in module.adc_gain_dict.items():
        adc_gain_old = adc_gain_old.data.item()
        adc_gain_new = adc_gain_old
        if adc_bit_old != module.adc_bit:
            adc_range_factor = 2 ** (module.adc_bit - adc_bit_old)
            adc_gain_new *= adc_range_factor

        if dac_bit_old != module.dac_bit:
            dac_range_factor = 2 ** (module.dac_bit - dac_bit_old)
            adc_gain_new /= dac_range_factor

        if weight_bit_old != module.weight_bit:
            weight_range_factor = 2 ** (module.weight_bit - weight_bit_old)
            adc_gain_new /= weight_range_factor

        adc_gain_new = torch.tensor(adc_gain_new, device = module.adc_gain.device)
        adc_gain_new = torch.clamp(adc_gain_new, min = 0.8 * module.adc_gain_min, max = 1.2 * module.adc_gain_max)
        module.adc_gain_dict[key].data = adc_gain_new

        if adc_gain_old != module.adc_gain_dict[key].data.item():
            print(f'adc_gain changed: {adc_gain_old} -> {adc_gain_new}')


def gen_adc_noise(module):
    # 增益噪声
    torch.manual_seed(module.seed)
    module.gain_noise = torch.randn(1000, device = module.weight.device) * module.gain_noise_scale
    # 偏置噪声
    module.offset_noise = torch.randn(1000, device = module.weight.device) * module.offset_noise_scale


def add_adc_noise(module, out_adc, start_col, col_num):
    if module.gain_noise_scale == 0 and module.offset_noise_scale == 0:
        return out_adc
    gain_noise_ = module.gain_noise[start_col:start_col + col_num]
    offset_noise_ = module.offset_noise[start_col:start_col + col_num]
    out_adc_gn = out_adc * (1 + gain_noise_)
    out_adc_off_n = out_adc_gn + module.adc_range * offset_noise_
    out_adc_off_n = (out_adc_off_n - out_adc).detach() + out_adc
    return out_adc_off_n


# def bit_split_tensor(x_q, x_bit, slice_bit):
#     assert slice_bit >= 1
#     bit_data_list = []
#     bit_len = int(math.ceil((x_bit - 1) / slice_bit))
#     for b in range(0, x_bit - 1, slice_bit):
#         lsb = b
#         msb = min(b + slice_bit, x_bit - 1)
#         shift_data = floor_pass(x_q / 2 ** lsb)
#         residue_data = floor_no_pass(x_q / 2 ** msb) * 2 ** slice_bit
#         bit_data = shift_data - residue_data
#         bit_data_pass = (bit_data - shift_data / bit_len).detach() + shift_data / bit_len
#         bit_data_list.append(bit_data_pass)
#     bit_data_list = torch.cat(bit_data_list, 0)
#     return bit_data_list


# def bit_concat_tensor(bitwise_data, data_bit, slice_bit):
#     bit_len = int(math.ceil((data_bit - 1) / slice_bit))
#     bitwise_data = torch.chunk(bitwise_data, chunks = bit_len, dim = 0)
#     data_sum = torch.zeros_like(bitwise_data[0], device = bitwise_data[0].device)
#     for i, bit_data in enumerate(bitwise_data):
#         data_sum += bit_data * 2 ** (i * slice_bit)
#     return data_sum

import math
def bit_concat_tensor(bitwise_data, data_bit, slice_bit):
    # 原实现：沿 dim=0 逐块 chunk 后 for 累加
    # 现在：一次性向量化加权求和（数值完全等价）
    bit_len = int(math.ceil((data_bit - 1) / slice_bit))
    # 期望 bitwise_data 的第 0 维是 bit_len * batch（与原实现保持一致）
    s0 = bitwise_data.shape[0]
    assert s0 % bit_len == 0, "bitwise_data 第一维必须能被 bit_len 整除"
    new_shape = (bit_len, s0 // bit_len) + tuple(bitwise_data.shape[1:])
    y = bitwise_data.reshape(new_shape)  # [bit_len, B, ...]
    # 2^(i*slice_bit) 的权重向量
    powv = (2 ** (slice_bit * torch.arange(bit_len, device=bitwise_data.device, dtype=bitwise_data.dtype)))
    # 广播到 y 的第 0 维，然后在 bit 维上求和
    # einsum 等价于: (y * powv[:, None, ...]).sum(dim=0)
    data_sum = torch.einsum('k...,...k->...', y, powv)  # 更稳：用下行等价写法
    # data_sum = (y * powv.view(bit_len, *([1] * (y.ndim - 1)))).sum(dim=0)
    return data_sum

def bit_split_tensor(x_q, x_bit, slice_bit):
    # 原实现：for b in range(bit_len) 逐块 floor/拼接
    # 现在：一次性向量化计算所有切片（保持与原实现数值完全一致，包括 msb 截断与 STE 规则）
    assert slice_bit >= 1
    bit_len = int(math.ceil((x_bit - 1) / slice_bit))
    # 构造 [bit_len] 的 lsb / msb 位移
    device = x_q.device
    dtype = x_q.dtype
    idx = torch.arange(bit_len, device=device, dtype=dtype)
    lsb = (idx * slice_bit)
    # msb 需要与原实现一致进行截断：min(lsb + slice_bit, x_bit - 1)
    msb = torch.minimum(lsb + slice_bit, torch.tensor(x_bit - 1, device=device, dtype=dtype))
    # 通过广播把 x_q 扩一维到 [bit_len, ...]
    xq_exp = x_q.unsqueeze(0)  # [1, ...] -> [bit_len, ...] 通过广播
    scale_lsb = (2 ** lsb).view(bit_len, *([1] * x_q.ndim))
    scale_msb = (2 ** msb).view(bit_len, *([1] * x_q.ndim))
    # 与原实现一致：shift_data 使用 floor_pass；residue_data 使用 floor_no_pass
    shift_data   = floor_pass(xq_exp / scale_lsb)
    residue_data = floor_no_pass(xq_exp / scale_msb) * (2 ** slice_bit)
    bit_data = shift_data - residue_data
    # 保持与原实现一致的直通梯度（对每个切片使用 shift_data/bit_len）
    bit_data_pass = (bit_data - shift_data / bit_len).detach() + (shift_data / bit_len)
    # 与原实现保持输出形状：沿第 0 维拼接 => [bit_len * B, ...]
    out = bit_data_pass.reshape(bit_len * x_q.shape[0], *x_q.shape[1:])
    return out
