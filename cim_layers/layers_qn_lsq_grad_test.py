# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import torch.nn.functional as F
from torch import nn

from cim_layers.quant_noise_utils import *


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
class Conv2d_qn_lsq(nn.Conv2d):
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

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

    def init_step_size(self, x, data_bit):
        _, scale = data_quant(x, data_bit = data_bit, isint = True)
        init_step_size = 1 / scale
        return init_step_size

    def update_para(self,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    gain_noise_scale = 0,
                    offset_noise_scale = 0,
                    ):
        if weight_bit is not None:
            self.weight_bit = weight_bit
            self.weight_half_level = 2 ** weight_bit / 2 - 1
        if input_bit is not None:
            self.input_bit = input_bit
            self.input_half_level = 2 ** input_bit / 2 - 1
        if output_bit is not None:
            self.output_bit = output_bit
            self.output_half_level = 2 ** output_bit / 2 - 1
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale

    def use_FP(self):
        return (self.weight_half_level <= 0 and
                self.input_half_level <= 0 and
                self.output_half_level <= 0 and
                self.noise_scale <= 0 and
                self.gain_noise_scale <= 0 and
                self.offset_noise_scale <= 0)

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
                                        isint = False)
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
                                          isint = False)
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

    def forward(self, x):
        if self.use_FP():
            x = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            x, in_scale = self.input_quant_noise(x)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            # ===================== #
            # 计算
            # ===================== #
            x = self._conv_forward(x, w_qn, bias = self.bias)
            # ===================== #
            # 输出量化
            # ===================== #
            x, out_scale = self.output_quant_noise(x)
        return x


# A fully connected layer which adds noise and quantize the weight and output feature map
# See notes in Conv2d_quant_noise
class Linear_qn_lsq(nn.Linear):
    def __init__(self, in_features, out_features,
                 weight_bit,
                 input_bit,
                 output_bit,
                 noise_scale,
                 input_quant = True,
                 output_quant = True,
                 weight_quant = True,
                 gain_noise_scale = 0,
                 offset_noise_scale = 0,
                 bias = False, ):
        super().__init__(in_features, out_features, bias)
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

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

    def init_step_size(self, x, data_bit):
        _, scale = data_quant(x, data_bit = data_bit, isint = True)
        init_step_size = 1 / scale
        return init_step_size

    def update_para(self,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    gain_noise_scale = 0,
                    offset_noise_scale = 0,
                    ):
        if weight_bit is not None:
            self.weight_bit = weight_bit
            self.weight_half_level = 2 ** weight_bit / 2 - 1
        if input_bit is not None:
            self.input_bit = input_bit
            self.input_half_level = 2 ** input_bit / 2 - 1
        if output_bit is not None:
            self.output_bit = output_bit
            self.output_half_level = 2 ** output_bit / 2 - 1
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale
    def use_FP(self):
        return (self.weight_half_level <= 0 and
                self.input_half_level <= 0 and
                self.output_half_level <= 0 and
                self.noise_scale <= 0 and
                self.gain_noise_scale <= 0 and
                self.offset_noise_scale <= 0)

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
                                        isint = False)
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
                                          isint = False)
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

    def forward(self, x):
        if self.use_FP():
            x = F.linear(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            x, in_scale = self.input_quant_noise(x)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            # ===================== #
            # 计算
            # ===================== #
            x = F.linear(x, w_qn, self.bias)
            # ===================== #
            # 输出量化
            # ===================== #
            x, out_scale = self.output_quant_noise(x)
        return x
