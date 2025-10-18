# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import math

import torch.nn.functional as F
from torch import nn

# from cim_layers.cim_layer_utils import *
from cim_layers.quant_noise_utils import *


# from memory_profiler import profile


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
class Conv2d_lsq_adda(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 weight_bit,
                 input_bit,
                 output_bit,
                 noise_scale,
                 dac_bit,
                 adc_bit,
                 adc_k,
                 input_quant = True,
                 output_quant = True,
                 weight_quant = True,
                 gain_noise_scale = 0,
                 offset_noise_scale = 0,
                 bias = True,
                 ):
        super().__init__(in_channels = in_channels,
                         out_channels = out_channels,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding,
                         groups = groups,
                         bias = bias,
                         )
        self.use_FP = False

        self.weight_bit = weight_bit
        self.weight_half_level = 2 ** weight_bit / 2 - 1

        self.input_bit = input_bit
        self.input_half_level = 2 ** input_bit / 2 - 1

        self.output_bit = output_bit
        self.output_half_level = 2 ** output_bit / 2 - 1

        self.noise_scale = noise_scale
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.weight_quant = weight_quant

        self.dac_bit = dac_bit
        self.adc_bit = adc_bit
        self.adc_k = adc_k

        self.slice_bit = dac_bit - 1
        self.bit_slices = int(math.ceil((input_bit - 1) / self.slice_bit))

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

        self.adc_gain = nn.Parameter(torch.tensor(1.0))

    def init_step_size(self, x, data_bit):
        _, scale = data_quant(x, data_bit = data_bit, isint = True)
        init_step_size = 1 / scale
        return init_step_size

    def update_step_size(self, weight_bit_old, input_bit_old, output_bit_old):
        if self.weight_bit != weight_bit_old:
            weight_step_size_factor = 2 ** (self.weight_bit - weight_bit_old)
            step_size_weight_old = self.step_size_weight.data.item()
            self.step_size_weight.data /= weight_step_size_factor
            print(f'step_size_weight changed: {step_size_weight_old} -> {self.step_size_weight.data}')

        if self.input_bit != input_bit_old:
            input_step_size_factor = 2 ** (self.input_bit - input_bit_old)
            step_size_input_old = self.step_size_input.data.item()
            self.step_size_input.data /= input_step_size_factor
            print(f'step_size_input changed: {step_size_input_old} -> {self.step_size_input.data}')

        if self.output_bit != output_bit_old:
            output_step_size_factor = 2 ** (self.output_bit - output_bit_old)
            step_size_output_old = self.step_size_output.data.item()
            self.step_size_output.data /= output_step_size_factor
            print(f'step_size_output changed: {step_size_output_old} -> {self.step_size_output.data}')

    def update_adc_gain(self, adc_bit_old, dac_bit_old, weight_bit_old):
        adc_gain_old = self.adc_gain.data.item()
        if adc_bit_old != self.adc_bit:
            adc_range_factor = 2 ** (self.adc_bit - adc_bit_old)
            adc_gain_new = max(self.adc_gain.data.item() * adc_range_factor, 1.0)
            self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if dac_bit_old != self.dac_bit:
            dac_range_factor = 2 ** (self.dac_bit - dac_bit_old)
            adc_gain_new = max(self.adc_gain.data.item() / dac_range_factor, 1.0)
            self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if weight_bit_old != self.weight_bit:
            weight_range_factor = 2 ** (self.weight_bit - weight_bit_old)
            adc_gain_new = max(self.adc_gain.data.item() / weight_range_factor, 1.0)
            self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if adc_gain_old != self.adc_gain.data.item():
            print(f'adc_gain changed: {adc_gain_old} -> {self.adc_gain.data.item()}')

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    adc_bit = None,
                    dac_bit = None,
                    gain_noise_scale = 0,
                    offset_noise_scale = 0,
                    ):
        weight_bit_old = self.weight_bit
        input_bit_old = self.input_bit
        output_bit_old = self.output_bit
        adc_bit_old = self.adc_bit
        dac_bit_old = self.dac_bit

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit
            self.weight_half_level = 2 ** weight_bit / 2 - 1

        if input_bit is not None:
            self.input_bit = input_bit
            self.input_half_level = 2 ** input_bit / 2 - 1
        if output_bit is not None:
            self.output_bit = output_bit
            self.output_half_level = 2 ** output_bit / 2 - 1

        if adc_bit is not None:
            self.adc_bit = adc_bit

        if dac_bit is not None:
            self.dac_bit = dac_bit

        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale

        self.update_adc_gain(adc_bit_old = adc_bit_old,
                             dac_bit_old = dac_bit_old,
                             weight_bit_old = weight_bit_old)

        self.update_step_size(weight_bit_old = weight_bit_old,
                              input_bit_old = input_bit_old,
                              output_bit_old = output_bit_old)

        self.slice_bit = self.dac_bit - 1
        self.bit_slices = int(math.ceil((self.input_bit - 1) / self.slice_bit))

    # @profile
    def input_quant_noise(self, x):
        x_scale = 1.0
        if self.input_quant:
            if self.step_size_input == 1:
                init_step_size = self.init_step_size(x, self.input_bit)
                self.step_size_input.data = init_step_size

                print(f'Initialized Step Size for Input: {self.step_size_input}')

            x, x_scale = data_quant_lsq(data_float = x,
                                        data_bit = self.input_bit,
                                        step_size = self.step_size_input,
                                        isint = True)
        return x, x_scale

    # @profile
    def weight_quant_noise(self):
        w_qn = self.weight
        w_scale = 1.0
        if self.weight_quant:
            if self.step_size_weight == 1:
                init_step_size = self.init_step_size(self.weight, self.weight_bit)
                self.step_size_weight.data = init_step_size
                print(f'Initialized Step Size for Weight: {self.step_size_weight}')

            w_q, w_scale = weight_quant_lsq(data_float = self.weight,
                                            data_bit = self.weight_bit,
                                            step_size = self.step_size_weight,
                                            isint = True)
            w_qn = add_noise(w_q, n_scale = self.noise_scale)
        return w_qn, w_scale

    # @profile
    def output_quant_noise(self, x):
        x_scale = 1.0
        if self.output_quant:
            if self.step_size_output == 1:
                init_step_size = self.init_step_size(x, self.output_bit)
                self.step_size_output.data = init_step_size

                print(f'Initialized Step Size for Output: {self.step_size_output}')
            x, x_scale = data_quant_lsq(data_float = x,
                                        data_bit = self.output_bit,
                                        step_size = self.step_size_output,
                                        isint = False)
        return x, x_scale

    def get_adc_scale(self):
        adc_gain = round_pass(self.adc_gain)
        adc_scale = adc_gain * self.adc_k
        return adc_scale

    def refresh_adc_params(self):
        self.adc_range = 2 ** (self.adc_bit - 1) - 1
        self.adc_scale = self.get_adc_scale()

    # @profile
    def forward(self, x):
        if self.use_FP:
            x = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            self.refresh_adc_params()
            # ===================== #
            # 输入量化
            # ===================== #
            x, in_scale = self.input_quant_noise(x)
            # ===================== #
            # 输入 bit 拆分
            # ===================== #
            x_list = bit_split_list(x_q = x,
                               x_bit = self.input_bit,
                               slice_bit = self.dac_bit - 1)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            # ===================== #
            # 计算
            # ===================== #
            out_list = []
            for x_ in x_list:
                out_ = self._conv_forward(x_, w_qn, bias = None)
                out_adc = self.adc_scale * out_
                out_adc = torch.clamp(out_adc, min = -self.adc_range - 1, max = self.adc_range)
                out_adc = round_pass(out_adc)
                out_list.append(out_adc)
            # ===================== #
            # 输出 bit 还原
            # ===================== #
            x = bit_concat_list(out_list, slice_bit = self.dac_bit - 1)
            # ===================== #
            # 量化系数还原
            # ===================== #
            x = x / w_scale / in_scale / self.adc_scale
            # ===================== #
            # 输出量化
            # ===================== #
            if self.bias is not None:
                x += self.bias.view(1, -1, 1, 1)
            x, out_scale = self.output_quant_noise(x)
        return x


class Linear_lsq_adda(nn.Linear):
    def __init__(self, in_features, out_features,
                 weight_bit,
                 input_bit,
                 output_bit,
                 noise_scale,
                 dac_bit,
                 adc_bit,
                 adc_k,
                 input_quant = True,
                 output_quant = True,
                 weight_quant = True,
                 gain_noise_scale = 0,
                 offset_noise_scale = 0,
                 bias = True, ):
        super().__init__(in_features, out_features, bias)
        self.use_FP = False

        self.weight_bit = weight_bit
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.input_bit = input_bit
        self.input_half_level = 2 ** input_bit / 2 - 1

        self.output_bit = output_bit
        self.output_half_level = 2 ** output_bit / 2 - 1

        self.noise_scale = noise_scale
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.weight_quant = weight_quant

        self.dac_bit = dac_bit
        self.adc_bit = adc_bit
        self.adc_k = adc_k

        self.slice_bit = dac_bit - 1
        self.bit_slices = int(math.ceil((input_bit - 1) / self.slice_bit))

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

        self.adc_gain = nn.Parameter(torch.tensor(1.0))

    def init_step_size(self, x, data_bit):
        _, scale = data_quant(x, data_bit = data_bit, isint = True)
        init_step_size = 1 / scale
        return init_step_size

    def update_step_size(self, weight_bit_old, input_bit_old, output_bit_old):
        if self.weight_bit != weight_bit_old:
            weight_step_size_factor = 2 ** (self.weight_bit - weight_bit_old)
            step_size_weight_old = self.step_size_weight.data.item()
            self.step_size_weight.data /= weight_step_size_factor
            print(f'step_size_weight changed: {step_size_weight_old} -> {self.step_size_weight.data}')

        if self.input_bit != input_bit_old:
            input_step_size_factor = 2 ** (self.input_bit - input_bit_old)
            step_size_input_old = self.step_size_input.data.item()
            self.step_size_input.data /= input_step_size_factor
            print(f'step_size_input changed: {step_size_input_old} -> {self.step_size_input.data}')

        if self.output_bit != output_bit_old:
            output_step_size_factor = 2 ** (self.output_bit - output_bit_old)
            step_size_output_old = self.step_size_output.data.item()
            self.step_size_output.data /= output_step_size_factor
            print(f'step_size_output changed: {step_size_output_old} -> {self.step_size_output.data}')

    def update_adc_gain(self, adc_bit_old, dac_bit_old, weight_bit_old):
        adc_gain_old = self.adc_gain.data.item()
        if adc_bit_old != self.adc_bit:
            adc_range_factor = 2 ** (self.adc_bit - adc_bit_old)
            adc_gain_new = max(self.adc_gain.data.item() * adc_range_factor, 1.0)
            self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if dac_bit_old != self.dac_bit:
            dac_range_factor = 2 ** (self.dac_bit - dac_bit_old)
            adc_gain_new = max(self.adc_gain.data.item() / dac_range_factor, 1.0)
            self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if weight_bit_old != self.weight_bit:
            weight_range_factor = 2 ** (self.weight_bit - weight_bit_old)
            adc_gain_new = max(self.adc_gain.data.item() / weight_range_factor, 1.0)
            self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if adc_gain_old != self.adc_gain.data.item():
            print(f'adc_gain changed: {adc_gain_old} -> {self.adc_gain.data.item()}')

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    adc_bit = None,
                    dac_bit = None,
                    gain_noise_scale = 0,
                    offset_noise_scale = 0,
                    ):
        weight_bit_old = self.weight_bit
        input_bit_old = self.input_bit
        output_bit_old = self.output_bit
        adc_bit_old = self.adc_bit
        dac_bit_old = self.dac_bit

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit
            self.weight_half_level = 2 ** weight_bit / 2 - 1

        if input_bit is not None:
            self.input_bit = input_bit
            self.input_half_level = 2 ** input_bit / 2 - 1
        if output_bit is not None:
            self.output_bit = output_bit
            self.output_half_level = 2 ** output_bit / 2 - 1

        if adc_bit is not None:
            self.adc_bit = adc_bit

        if dac_bit is not None:
            self.dac_bit = dac_bit

        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale

        self.update_adc_gain(adc_bit_old = adc_bit_old,
                             dac_bit_old = dac_bit_old,
                             weight_bit_old = weight_bit_old)

        self.update_step_size(weight_bit_old = weight_bit_old,
                              input_bit_old = input_bit_old,
                              output_bit_old = output_bit_old)

        self.slice_bit = self.dac_bit - 1
        self.bit_slices = int(math.ceil((self.input_bit - 1) / self.slice_bit))


    def input_quant_noise(self, x):
        x_scale = 1.0
        if self.input_quant:
            if self.step_size_input == 1:
                init_step_size = self.init_step_size(x, self.input_bit)
                self.step_size_input.data = init_step_size

                print(f'Initialized Step Size for Input: {self.step_size_input}')

            x, x_scale = data_quant_lsq(data_float = x,
                                        data_bit = self.input_bit,
                                        step_size = self.step_size_input,
                                        isint = True)
        return x, x_scale

    def weight_quant_noise(self):
        w_qn = self.weight
        w_scale = 1.0
        if self.weight_quant:
            if self.step_size_weight == 1:
                init_step_size = self.init_step_size(self.weight, self.weight_bit)
                self.step_size_weight.data = init_step_size
                print(f'Initialized Step Size for Weight: {self.step_size_weight}')

            w_q, w_scale = weight_quant_lsq(data_float = self.weight,
                                            data_bit = self.weight_bit,
                                            step_size = self.step_size_weight,
                                            isint = True)
            w_qn = add_noise(w_q, n_scale = self.noise_scale)
        return w_qn, w_scale

    def output_quant_noise(self, x):
        x_scale = 1.0
        if self.output_quant:
            if self.step_size_output == 1:
                init_step_size = self.init_step_size(x, self.output_bit)
                self.step_size_output.data = init_step_size

                print(f'Initialized Step Size for Output: {self.step_size_output}')
            x, x_scale = data_quant_lsq(data_float = x,
                                        data_bit = self.output_bit,
                                        step_size = self.step_size_output,
                                        isint = False)
        return x, x_scale

    def get_adc_scale(self):
        adc_gain = round_pass(self.adc_gain)
        adc_scale = adc_gain * self.adc_k
        return adc_scale

    def refresh_adc_params(self):
        self.adc_range = 2 ** (self.adc_bit - 1) - 1
        self.adc_scale = self.get_adc_scale()

    def forward(self, x):
        if self.use_FP:
            x = F.linear(x, self.weight, bias = self.bias)

        else:
            self.refresh_adc_params()
            # ===================== #
            # 输入量化
            # ===================== #
            x, in_scale = self.input_quant_noise(x)
            # ===================== #
            # 输入 bit 拆分
            # ===================== #
            x_list = bit_split_list(x_q = x, x_bit = self.input_bit, slice_bit = self.dac_bit - 1)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            # ===================== #
            # 计算
            # ===================== #
            out_list = []
            for x_ in x_list:
                out_ = F.linear(x_, w_qn, bias = None)
                out_adc = self.adc_scale * out_
                out_adc = torch.clamp(out_adc, min = -self.adc_range - 1, max = self.adc_range)
                out_adc = round_pass(out_adc)
                out_list.append(out_adc)
            # ===================== #
            # 输出 bit 还原
            # ===================== #
            x = bit_concat_list(out_list, slice_bit = self.dac_bit - 1)
            # ===================== #
            # 量化系数还原
            # ===================== #
            x = x / w_scale / in_scale / self.adc_scale
            # ===================== #
            # 输出量化
            # ===================== #
            if self.bias is not None:
                x += self.bias.view(1, -1)
            x, out_scale = self.output_quant_noise(x)
        return x


def bit_split_list(x_q, x_bit, slice_bit):
    assert slice_bit >= 1
    bit_data_list = []
    bit_len = int(math.ceil((x_bit - 1) / slice_bit))
    for b in range(0, x_bit - 1, slice_bit):
        lsb = b
        msb = min(b + slice_bit, x_bit - 1)
        shift_data = floor_pass(x_q / 2 ** lsb)
        residue_data = floor_no_pass(x_q / 2 ** msb) * 2 ** slice_bit
        bit_data = shift_data - residue_data
        bit_data_pass = (bit_data - shift_data / bit_len).detach() + shift_data / bit_len
        bit_data_list.append(bit_data_pass)
    return bit_data_list


def bit_concat_list(bit_data_list, slice_bit):
    data_sum = torch.zeros_like(bit_data_list[0], device = bit_data_list[0].device)
    for i, bit_data in enumerate(bit_data_list):
        data_sum += bit_data * 2 ** (i * slice_bit)
    return data_sum
