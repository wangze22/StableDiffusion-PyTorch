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
from cim_layers.custom_modules import *
from cim_toolchain_utils.utils import scatter_plt


# torch.manual_seed(0)


# ====================== #
# 基于全整型的神经网络
# ====================== #
class Conv2d_int(nn.Module):
    def __init__(self,
                 stride,
                 padding,
                 groups,
                 # output_bit,
                 ):
        super().__init__()

        # self.output_bit = output_bit
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x_int, weight_int, bias_int):
        y_int = F.conv2d(x_int,
                         weight = weight_int,
                         bias = bias_int,
                         stride = self.stride,
                         padding = self.padding,
                         groups = self.groups)
        return y_int


class Conv2d_lsq_int(nn.Module):
    def __init__(self, conv_lsq,
                 int_grad = False, weight_bit_extension = 4):
        super().__init__()
        # 是否采用整型反传训练
        # 整型反传训练是用来仿真片上训练收敛情况的
        self.int_grad = int_grad
        self.weight_bit_extension = weight_bit_extension
        self.extended_levels = 2 ** self.weight_bit_extension

        # 提取网络配置
        self.in_channels = conv_lsq.in_channels
        self.out_channels = conv_lsq.out_channels
        self.stride = conv_lsq.stride
        self.padding = conv_lsq.padding
        self.groups = conv_lsq.groups
        self.bias = conv_lsq.bias

        # bit 精度定义
        self.weight_bit = conv_lsq.weight_bit
        self.input_bit = conv_lsq.input_bit
        self.output_bit = conv_lsq.output_bit
        self.bias_bit = conv_lsq.output_bit
        self.output_range = 2 ** (self.output_bit - 1) - 1

        # 提取 step size
        step_size_input = conv_lsq.step_size_input.data.clone()
        step_size_output = conv_lsq.step_size_output.data.clone()
        step_size_weight = conv_lsq.step_size_weight.data.clone()

        # 提取浮点数权重值
        self.weight = nn.Parameter(conv_lsq.weight.data.clone())

        if self.bias is not None:
            self.bias = nn.Parameter(conv_lsq.bias.data.clone())
            self.lsq_quant_bias = Quant_layer(isint = False,
                                              data_bit = self.output_bit)

        # 定义 LSQ 量化和反量化层
        self.lsq_quant_input = Quant_layer(isint = True, data_bit = self.input_bit, step_size = step_size_input)
        self.lsq_quant_weight = Quant_layer(isint = True, data_bit = self.weight_bit, step_size = step_size_weight)
        self.mvm_result_shift = Bit_shift_layer(data_bit = self.output_bit)
        self.lsq_quant_output = Quant_layer(isint = False, data_bit = self.output_bit, step_size = step_size_output)

        # 定义整数卷积层
        self.conv2d_int = Conv2d_int(stride = self.stride,
                                     padding = self.padding,
                                     groups = self.groups)

        # 训练追踪层
        self.identity = Identity_layer()
        self.weight_to_int_flag = False
        # 如果采用整型训练，则固定 weight 和 bias 的 step_size
        if self.int_grad:
            self.use_int_grad(weight_bit_extension)

    def use_int_grad(self, weight_bit_extension):
        # 是否采用整型反传训练
        self.int_grad = True
        self.weight_bit_extension = weight_bit_extension
        self.extended_levels = 2 ** self.weight_bit_extension

        # 如果采用整型反传训练，固定住 step_size
        # 只训练 weight 和 bias
        self.lsq_quant_weight.step_size.requires_grad = False
        self.lsq_quant_input.step_size.requires_grad = False
        self.lsq_quant_output.step_size.requires_grad = False
        self.mvm_result_shift.step_size.requires_grad = False
        if hasattr(self, 'lsq_quant_bias'):
            self.lsq_quant_bias.step_size.requires_grad = False

    def gen_ext_weight(self):
        step_size_weight = self.lsq_quant_weight.step_size
        w_step_size = step_size_weight / self.extended_levels
        w_data_int, w_scale = weight_quant_round(data_float = self.weight.detach().clone(),
                                                 data_bit = self.weight_bit + self.weight_bit_extension,
                                                 step_size = w_step_size,
                                                 isint = 1)
        self.weight_to_int_flag = True
        self.weight.data = w_data_int
        self.weight_scale_ext = w_scale
        self.weight_scale = 1 / step_size_weight

    def weight_quant_fine(self):
        weight = floor_no_pass(self.weight.data)
        self.weight.data = weight

    def grad_scale(self, weight, scale):
        weight = weight * scale - (weight * scale).detach() + weight.detach()
        return weight

    def weight_quant_forward(self):
        weight = self.weight
        # weight = self.grad_scale(weight, scale = self.weight_scale_ext)
        weight = self.grad_scale(weight, scale = self.extended_levels)
        w_data_int = weight / self.extended_levels
        w_data_int = floor_pass(w_data_int)
        return w_data_int

    def bias_quant_fine(self):
        step_size_bias = self.lsq_quant_bias.step_size
        bias_data_int, _ = weight_quant_floor(data_float = self.bias,
                                              data_bit = self.output_bit + self.weight_bit_extension,
                                              step_size = step_size_bias / self.extended_levels,
                                              isint = 0)
        self.bias.data = bias_data_int

    def bias_quant_forward(self):
        step_size_bias = self.lsq_quant_bias.step_size
        w_data_int, w_scale = weight_quant_floor(data_float = self.bias,
                                                 data_bit = self.output_bit,
                                                 step_size = step_size_bias,
                                                 isint = 0)
        return w_data_int, w_scale

    def forward(self, x):
        # ========================== #
        # DDFP 量化: FP -> INT
        # ========================== #
        if self.int_grad:
            x_int, x_scale = self.lsq_quant_input(x.detach())
            # ------------------------- #
            # 采用整型反传，需要每次foward时把权重数据量化为 n+extend 比特
            # ------------------------- #
            if not self.weight_to_int_flag:
                self.gen_ext_weight()
            self.weight_quant_fine()
            w_int = self.weight_quant_forward()
            w_scale = self.weight_scale
        else:
            x_int, x_scale = self.lsq_quant_input(x)
            w_int, w_scale = self.lsq_quant_weight(self.weight)
        y_int = self.conv2d_int(x_int = x_int,
                                weight_int = w_int,
                                bias_int = None
                                )
        self.y_int = y_int
        self.y_int.retain_grad()
        # MVM 结果通过移位变为 8 bit
        y_shift, y_shift_scale = self.mvm_result_shift(y_int)
        self.y_shift = y_shift
        self.y_shift.retain_grad()
        # MVM 结果加上 bias
        if self.bias is not None:
            if self.int_grad:
                self.bias_quant_fine()
                b_int, b_scale = self.bias_quant_forward()
            else:
                b_int, b_scale = self.lsq_quant_bias(self.bias)

            b_int8 = round_pass(b_int * y_shift_scale * x_scale * w_scale)

            # self.b_int8 = b_int8
            # self.b_int8.retain_grad()

            b_int8 = torch.clamp(b_int8, -self.output_range, self.output_range)
            y_shift = y_shift + b_int8.view(1, -1, 1, 1)
            y_shift = torch.clamp(y_shift, -self.output_range, self.output_range)

        y_shift_DMAC = self.identity(y_shift)
        self.y_shift_DMAC = y_shift_DMAC
        self.y_shift_DMAC.retain_grad()
        # ========================== #
        # DDFP 反量化: INT -> FP
        # ========================== #
        y_shift = y_shift_DMAC / x_scale / w_scale / y_shift_scale
        y_fp, y_fp_scale = self.lsq_quant_output(y_shift)

        return y_shift_DMAC


class Linear_lsq_int(nn.Module):
    def __init__(self, linear_lsq):
        super().__init__()
        print(f'没有定义 Linear LSQ INT 层')
        exit(1)


def check_integer(tensor, tolerance = 1e-10):
    is_integer_tensor = torch.all((tensor - tensor.round()).abs() < tolerance)
    return is_integer_tensor


def weight_quant_int(module):
    if check_integer(module.weight, tolerance = 1e-10):
        return
    if module.step_size_weight == 1:
        module.step_size_weight.data = init_step_size(module.weight, module.weight_bit)
        print(f'Initialized Step Size for Weight: {module.step_size_weight.data.item()}')

    step_size = module.step_size_weight
    data_float = module.weight
    data_bit = module.weight_bit

    data_range = 2 ** (data_bit - 1) - 1
    data_scaled = data_float / step_size
    data_clamped = torch.clamp(data_scaled, -data_range, data_range)
    data_quantized = round_pass(data_clamped)

    module.weight.data = data_quantized
    return


def bias_quant_int(module):
    if check_integer(module.bias, tolerance = 1e-10):
        return

    step_size_weight = module.step_size_weight
    step_size_input = module.step_size_input
    bias_data = module.bias.data.detach()

    bias_bit = 16
    bias_range = 2 ** (bias_bit - 1) - 1

    data_scaled = bias_data / step_size_weight / step_size_input
    data_clamped = torch.clamp(data_scaled, -bias_range, bias_range)
    data_quantized = round_pass(data_clamped)

    module.bias.data = data_quantized
    return


def weight_clamp(module):
    weight = module.weight
    data_bit = module.weight_bit
    data_range = 2 ** (data_bit - 1) - 1
    weight_clamped = torch.clamp(weight, -data_range, data_range)
    module.weight.data = weight_clamped


def bias_clamp(module):
    bias = module.bias

    bias_bit = 16
    bias_range = 2 ** (bias_bit - 1) - 1

    bias_clamped = torch.clamp(bias, -bias_range, bias_range)
    module.bias.data = bias_clamped


def unique_values(tensor):
    unique_values = torch.unique(tensor)
    return unique_values.numel()  # 返回唯一值的数量


def count_abnormal_values(tensor):
    # 检查 NaN 和 Inf 元素，并统计其数量
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    # 返回异常值总数
    return nan_count + inf_count


def find_abnormal_values(tensor):
    # 找出 NaN 和 Inf 元素的位置
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    # 获取异常值的数量和位置
    total_count = (nan_mask | inf_mask).sum().item()
    abnormal_positions = (nan_mask | inf_mask).nonzero(as_tuple = True)

    return total_count, abnormal_positions
