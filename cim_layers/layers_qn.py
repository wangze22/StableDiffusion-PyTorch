# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import torch.nn.functional as F
from torch import nn

from cim_layers.quant_noise_utils import *

class Conv2d_quant_noise(nn.Conv2d):
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
                 clamp_std,
                 noise_scale,
                 input_quant = True,
                 output_quant = True,
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
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale
        self.weight_bit = weight_bit

        self.input_bit = input_bit

        self.output_bit = output_bit

        self.input_quant = input_quant
        self.output_quant = output_quant

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    clamp_std = None,
                    noise_scale = None,
                    gain_noise_scale = None,
                    offset_noise_scale = None,
                    ):

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit

        if input_bit is not None:
            self.input_bit = input_bit

        if output_bit is not None:
            self.output_bit = output_bit

        if clamp_std is not None:
            self.clamp_std = clamp_std
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale


    def forward(self, x):
        if self.use_FP:
            x = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            if self.input_quant:
                x, x_scale = data_quant_pass(x, data_bit = self.input_bit, isint = False)
            # ===================== #
            # 权重截断&量化
            # ===================== #
            # w_q 对应权重的写校验值
            w_std = self.weight.std()
            if self.clamp_std > 0:
                w_c = torch.clamp(self.weight, -w_std * self.clamp_std, w_std * self.clamp_std)
            else:
                w_c = self.weight + 0
            w_q, w_scale = data_quant_pass(w_c, data_bit = self.weight_bit, isint = False)
            w_qn = add_noise(w_q, n_scale = self.noise_scale)

            # calculate the convolution next
            x = self._conv_forward(x, w_qn, bias = self.bias)

            # quantize the output feature map at last
            if self.output_quant:
                x, x_scale = data_quant_pass(x, data_bit = self.output_bit, isint = False)
        return x


# A fully connected layer which adds noise and quantize the weight and output feature map
# See notes in Conv2d_quant_noise
class Linear_quant_noise(nn.Linear):
    def __init__(self, in_features, out_features,
                 weight_bit,
                 input_bit,
                 output_bit,
                 clamp_std,
                 noise_scale,
                 input_quant = True,
                 output_quant = True,
                 gain_noise_scale = 0,
                 offset_noise_scale = 0,
                 bias = False, ):
        super().__init__(in_features, out_features, bias)
        self.use_FP = False

        self.weight_bit = weight_bit

        self.input_bit = input_bit

        self.output_bit = output_bit

        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale
        self.weight_bit = weight_bit

        self.input_quant = input_quant
        self.output_quant = output_quant

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    clamp_std = None,
                    noise_scale = None,
                    gain_noise_scale = None,
                    offset_noise_scale = None,
                    ):

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit

        if input_bit is not None:
            self.input_bit = input_bit

        if output_bit is not None:
            self.output_bit = output_bit

        if clamp_std is not None:
            self.clamp_std = clamp_std
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale


    def forward(self, x):
        if self.use_FP:
            x = F.linear(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            if self.input_quant:
                x, x_scale = data_quant_pass(x, data_bit = self.input_bit, isint = False)
            # ===================== #
            # 权重截断&量化
            # ===================== #
            # w_q 对应权重的写校验值
            w_std = self.weight.std()
            if self.clamp_std > 0:
                w_c = torch.clamp(self.weight, -w_std * self.clamp_std, w_std * self.clamp_std)
            else:
                w_c = self.weight + 0
            w_q, w_scale = data_quant_pass(w_c, data_bit = self.weight_bit, isint = False)
            w_qn = add_noise(w_q, n_scale = self.noise_scale)

            x = F.linear(x, w_qn, self.bias)

            if self.output_quant:
                x, x_scale = data_quant_pass(x, data_bit = self.output_bit, isint = False)
        return x