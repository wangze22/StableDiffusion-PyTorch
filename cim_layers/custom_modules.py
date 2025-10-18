# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import torch.nn.functional
import torch.nn.functional as F
from torch import nn

from cim_layers.quant_noise_utils import *
from cim_layers.layers_utils_lsq import *
from cim_toolchain_utils.utils import scatter_plt


# ====================================================================== #
# 其他自定义工具层，继承自 nn.Module
# ====================================================================== #
class Quant_layer(nn.Module):
    def __init__(self, isint, data_bit, step_size = None):
        super().__init__()
        if step_size is None:
            self.step_size = nn.Parameter(torch.tensor(1.0))
        else:
            self.step_size = nn.Parameter(step_size)
        self.isint = isint
        self.data_bit = data_bit

    def lsq_quant(self, x):
        if self.step_size == 1:
            self.step_size.data = init_step_size(x, self.data_bit)
            print(f'Initialized Step Size for Quant Layer: {self.step_size.data.item()}')

        x, x_scale = data_quant_lsq(data_float = x,
                                    data_bit = self.data_bit,
                                    step_size = self.step_size,
                                    isint = self.isint)
        return x, x_scale

    def forward(self, x):
        x_q, x_scale = self.lsq_quant(x)
        return x_q, x_scale


class Bit_shift_layer(nn.Module):
    def __init__(self, data_bit, step_size = None):
        super().__init__()
        if step_size is None:
            self.step_size = nn.Parameter(torch.tensor(1.0))
        else:
            self.step_size = nn.Parameter(step_size)
        self.data_bit = data_bit

    def data_quant_floor(self, data_float, data_bit, step_size):
        assert data_bit > 0
        quant_scale = (1 / step_size).detach()

        data_range = 2 ** (data_bit - 1) - 1
        gradScaleFactor = grad_scale_factor(data_range, data_float)

        step_size = grad_scale(step_size, gradScaleFactor)

        data_scaled = data_float / step_size
        data_clamped = torch.clamp(data_scaled, -data_range, data_range)

        data_quantized = floor_pass(data_clamped)

        data_quantized = data_quantized * step_size / step_size.detach()
        return data_quantized, quant_scale

    def bit_shift(self, x):
        if self.step_size == 1:
            self.step_size.data = init_step_size(x, self.data_bit)
            print(f'Initialized Step Size for Bit Shift Layer: {self.step_size.data.item()}')

        shift_bits = torch.log2(self.step_size)
        shift_bits = round_pass_exp(shift_bits)
        # shift_bits = round_pass(shift_bits)
        x, x_scale = self.data_quant_floor(data_float = x,
                                           data_bit = self.data_bit,
                                           step_size = 2 ** shift_bits)
        return x, x_scale

    @property
    def shift_bits(self):
        shift_bits = torch.log2(self.step_size)
        shift_bits = round_pass(shift_bits)
        return shift_bits

    def forward(self, x):
        x_q, x_scale = self.bit_shift(x)
        return x_q, x_scale


class Identity_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_flag = 'DMAC_train'

    def forward(self, x):
        return x
