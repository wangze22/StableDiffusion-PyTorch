# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import torch.nn.functional as F
from torch import nn

from cim_layers.quant_noise_utils import *
from cim_layers.layers_utils_lsq import *


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
        self.use_FP = False

        self.weight_bit = weight_bit

        self.input_bit = input_bit

        self.output_bit = output_bit

        self.noise_scale = noise_scale
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.weight_quant = weight_quant

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    gain_noise_scale = 0,
                    offset_noise_scale = 0,
                    ):

        weight_bit_old = self.weight_bit
        input_bit_old = self.input_bit
        output_bit_old = self.output_bit

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit

        if input_bit is not None:
            self.input_bit = input_bit

        if output_bit is not None:
            self.output_bit = output_bit

        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale

        update_step_size(self,
                         weight_bit_old = weight_bit_old,
                         input_bit_old = input_bit_old,
                         output_bit_old = output_bit_old)

    def forward(self, x):
        if self.use_FP:
            x = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            x, in_scale = input_quant(self, x, isint = False)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = weight_quant_noise(self, isint = False)
            # ===================== #
            # 计算
            # ===================== #
            x = self._conv_forward(x, w_qn, bias = self.bias)
            # ===================== #
            # 输出量化
            # ===================== #
            x, out_scale = output_quant(self, x, isint = False)
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
        self.use_FP = False

        self.weight_bit = weight_bit

        self.input_bit = input_bit

        self.output_bit = output_bit

        self.noise_scale = noise_scale
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.weight_quant = weight_quant

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    gain_noise_scale = 0,
                    offset_noise_scale = 0,
                    ):

        weight_bit_old = self.weight_bit
        input_bit_old = self.input_bit
        output_bit_old = self.output_bit

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit

        if input_bit is not None:
            self.input_bit = input_bit

        if output_bit is not None:
            self.output_bit = output_bit

        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale

        update_step_size(self,
                         weight_bit_old = weight_bit_old,
                         input_bit_old = input_bit_old,
                         output_bit_old = output_bit_old)

    def forward(self, x):
        if self.use_FP:
            x = F.linear(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            x, in_scale = input_quant(self, x, isint = False)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = weight_quant_noise(self, isint = False)
            # ===================== #
            # 计算
            # ===================== #
            x = F.linear(x, w_qn, self.bias)
            # ===================== #
            # 输出量化
            # ===================== #
            x, out_scale = output_quant(self, x, isint = False)
        return x
