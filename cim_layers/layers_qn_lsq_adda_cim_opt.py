# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

# from cim_layers.cim_layer_utils import *
from cim_layers.quant_noise_utils import *
from cim_layers.layers_utils_lsq import *
from cim_layers.layers_utils_adda import *
try:
    from cim_layers.bitsplit import bitsplit_ext
except:
    pass
# from memory_profiler import profile


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
class Conv2d_lsq_adda_cim(nn.Conv2d):
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
                 adc_gain_1_scale,
                 adc_gain_range,
                 adc_adjust_mode = 'gain',
                 input_quant = True,
                 output_quant = True,
                 weight_quant = True,
                 gain_noise_scale = 0,
                 offset_noise_scale = 0,
                 seed = 0,
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

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.weight_quant = weight_quant

        self.dac_bit = dac_bit
        self.adc_bit = adc_bit
        self.adc_gain_1_scale = adc_gain_1_scale

        self.slice_bit = dac_bit - 1
        self.bit_slices = int(math.ceil((input_bit - 1) / self.slice_bit))

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

        # ===================================== #
        # ADC 增益调整参数
        # ===================================== #
        # ADC 增益初始化很重要，如果初始化为 1.0， 在ResNet的训练中会出现梯度为NaN的情况
        self.adc_gain_min = float(adc_gain_range[0])
        self.adc_gain_max = float(adc_gain_range[1])
        self.adc_gain = nn.Parameter(torch.tensor(self.adc_gain_min))
        self.adc_adjust_mode = adc_adjust_mode

        self.unfold = nn.Unfold(kernel_size = kernel_size,
                                padding = padding,
                                stride = stride)
        self.shape_info = None

        # ===================================== #
        # ADC 噪声
        # ===================================== #
        self.seed = seed
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale
        gen_adc_noise(self)

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    adc_bit = None,
                    dac_bit = None,
                    gain_noise_scale = None,
                    offset_noise_scale = None,
                    seed = None,
                    ):
        weight_bit_old = self.weight_bit
        input_bit_old = self.input_bit
        output_bit_old = self.output_bit
        adc_bit_old = self.adc_bit
        dac_bit_old = self.dac_bit

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit

        if input_bit is not None:
            self.input_bit = input_bit

        if output_bit is not None:
            self.output_bit = output_bit

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
        if seed is not None:
            self.seed = seed
        gen_adc_noise(self)
        update_adc_gain(self,
                        adc_bit_old = adc_bit_old,
                        dac_bit_old = dac_bit_old,
                        weight_bit_old = weight_bit_old)

        update_step_size(self,
                         weight_bit_old = weight_bit_old,
                         input_bit_old = input_bit_old,
                         output_bit_old = output_bit_old)

        self.slice_bit = self.dac_bit - 1
        self.bit_slices = int(math.ceil((self.input_bit - 1) / self.slice_bit))

    # @profile
    def get_shape_info(self, x):
        self.shape_info = x.shape
        input_rows = self.shape_info[2]
        input_cols = self.shape_info[3]
        self.out_rows = int((input_rows + self.padding[0] * 2 - self.kernel_size[0]) / self.stride[0] + 1)
        self.out_cols = int((input_cols + self.padding[0] * 2 - self.kernel_size[0]) / self.stride[0] + 1)

    # @profile
    def fold_output(self, x):
        batch = x.shape[0]
        x = x.permute(0, 2, 1).reshape([batch,
                                        self.out_channels,
                                        self.out_rows, self.out_cols])
        return x

    # @profile
    def get_2d_weight(self, weight):
        weight_2d = weight.reshape(self.out_channels, -1).transpose(0, 1)
        return weight_2d

    # @profile
    def gen_output_tensor(self, x_2d):
        batch_num = x_2d.shape[0] // self.bit_slices
        output_rows = x_2d.shape[2]
        output_concated = torch.zeros([batch_num,
                                       output_rows,
                                       self.out_channels],
                                      device = self.weight.device)
        return output_concated

    # @profile
    def cal_x_weight_block(self, x_expanded, w_qn):
        # ===================== #
        # 权重 输入 2D 展开
        # ===================== #
        w_qn = self.get_2d_weight(w_qn)
        x_q_2d = self.unfold(x_expanded)
        # ===================== #
        # 生成 2D 计算结果矩阵
        # ===================== #
        output_concat = self.gen_output_tensor(x_q_2d)
        # ===================== #
        # 逐个 mapping 位置计算
        # ===================== #
        for key, weight_info in self.weight_mapping_info.items():
            start_row = weight_info['start_row']
            start_col = weight_info['start_col']
            row_num = weight_info['row_num']
            col_num = weight_info['col_num']
            # ===================== #
            # 输入数据提取
            # ===================== #
            x_split = x_q_2d[:, start_row:start_row + row_num, :]
            # ===================== #
            # 权重提取
            # ===================== #
            w_qn_in = w_qn[start_row:start_row + row_num, start_col:start_col + col_num]
            # ===================== #
            # 逐 bit 计算
            # ===================== #
            output_bitwise = self.cal_x_bitwise(x_split, w_qn_in, start_col, col_num)
            # ===================== #
            # 输出 bit 还原
            # ===================== #
            x = bit_concat_tensor(output_bitwise, data_bit = self.input_bit, slice_bit = self.slice_bit)
            # ===================== #
            # 结果对应位置累加
            # ===================== #
            output_concat[:, :, start_col:start_col + col_num] += x
        return output_concat

    # @profile
    def cal_x_bitwise(self, x_expanded, w_qn, start_col, col_num):
        bit_len_batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        x_cols = x_expanded.shape[2]
        out_ = torch.matmul(x_expanded.permute(0, 2, 1), w_qn)
        if self.adc_gain == self.adc_gain_min:
            init_adc_gain(self, out_, self.adc_adjust_mode)
        out_adc = self.adc_scale * out_
        out_adc = add_adc_noise(self, out_adc, start_col, col_num)
        out_adc = torch.clamp(out_adc, min = -self.adc_range - 1, max = self.adc_range)
        out_adc = round_pass(out_adc)
        return out_adc

    # @profile
    def refresh_adc_params(self):
        self.adc_range = 2 ** (self.adc_bit - 1) - 1
        self.adc_scale = get_adc_scale(self, self.adc_gain, mode = self.adc_adjust_mode)

    # @profile
    def forward(self, x):
        if self.use_FP:
            x = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            if self.shape_info is None:
                self.get_shape_info(x)
            self.refresh_adc_params()
            # ===================== #
            # 输入量化
            # ===================== #
            x_q, in_scale = input_quant(self, x, isint = True)
            # ===================== #
            # 输入 bit 拆分
            # ===================== #
            x_expanded = bitsplit_ext.bit_split(x_q = x_q,
                                          x_bit = self.input_bit,
                                          slice_bit = self.slice_bit)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = weight_quant_noise(self, isint = True)
            # ===================== #
            # 计算
            # ===================== #
            output_concated = self.cal_x_weight_block(x_expanded, w_qn)
            x = self.fold_output(output_concated)
            # ===================== #
            # 量化系数还原
            # ===================== #
            x = x / w_scale / in_scale / self.adc_scale
            # ===================== #
            # 输出量化
            # ===================== #
            if self.bias is not None:
                x += self.bias.view(1, -1, 1, 1)
            x, out_scale = output_quant(self, x, isint = False)
        return x


class Linear_lsq_adda_cim(nn.Linear):
    def __init__(self, in_features, out_features,
                 weight_bit,
                 input_bit,
                 output_bit,
                 noise_scale,
                 dac_bit,
                 adc_bit,
                 adc_gain_1_scale,
                 adc_gain_range,
                 adc_adjust_mode = 'gain',
                 input_quant = True,
                 output_quant = True,
                 weight_quant = True,
                 gain_noise_scale = 0,
                 offset_noise_scale = 0,
                 seed = 0,
                 bias = True, ):
        super().__init__(in_features, out_features, bias)
        self.use_FP = False

        self.weight_bit = weight_bit

        self.input_bit = input_bit

        self.output_bit = output_bit

        self.noise_scale = noise_scale

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.weight_quant = weight_quant

        self.dac_bit = dac_bit
        self.adc_bit = adc_bit
        self.adc_gain_1_scale = adc_gain_1_scale

        self.slice_bit = dac_bit - 1
        self.bit_slices = int(math.ceil((input_bit - 1) / self.slice_bit))

        self.step_size_input = nn.Parameter(torch.tensor(1.0))
        self.step_size_output = nn.Parameter(torch.tensor(1.0))
        self.step_size_weight = nn.Parameter(torch.tensor(1.0))

        # ===================================== #
        # ADC 增益调整参数
        # ===================================== #
        # ADC 增益初始化很重要，如果初始化为 1.0， 在ResNet的训练中会出现梯度为NaN的情况
        self.adc_gain_min = float(adc_gain_range[0])
        self.adc_gain_max = float(adc_gain_range[1])
        self.adc_gain = nn.Parameter(torch.tensor(self.adc_gain_min))
        self.adc_adjust_mode = adc_adjust_mode
        # ===================================== #
        # ADC 噪声
        # ===================================== #
        self.seed = seed
        self.gain_noise_scale = gain_noise_scale
        self.offset_noise_scale = offset_noise_scale
        gen_adc_noise(self)

    def update_para(self,
                    use_FP = False,
                    weight_bit = None,
                    input_bit = None,
                    output_bit = None,
                    noise_scale = None,
                    adc_bit = None,
                    dac_bit = None,
                    gain_noise_scale = None,
                    offset_noise_scale = None,
                    seed = None,
                    ):
        weight_bit_old = self.weight_bit
        input_bit_old = self.input_bit
        output_bit_old = self.output_bit
        adc_bit_old = self.adc_bit
        dac_bit_old = self.dac_bit

        self.use_FP = use_FP
        if weight_bit is not None:
            self.weight_bit = weight_bit

        if input_bit is not None:
            self.input_bit = input_bit

        if output_bit is not None:
            self.output_bit = output_bit

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
        if seed is not None:
            self.seed = seed
        gen_adc_noise(self)
        update_adc_gain(self,
                        adc_bit_old = adc_bit_old,
                        dac_bit_old = dac_bit_old,
                        weight_bit_old = weight_bit_old)

        update_step_size(self,
                         weight_bit_old = weight_bit_old,
                         input_bit_old = input_bit_old,
                         output_bit_old = output_bit_old)

        self.slice_bit = self.dac_bit - 1
        self.bit_slices = int(math.ceil((self.input_bit - 1) / self.slice_bit))

    # @profile
    def gen_output_tensor(self, x_2d):
        # Support N-D inputs: x_2d has shape [bit_len * B, ..., in_features]
        # We need to recover batch dim B and preserve any extra leading dims
        # so that the final output matches nn.Linear: [..., out_features]
        assert x_2d.shape[-1] == self.in_features, "Last dimension must equal in_features"
        leading_dims = list(x_2d.shape[:-1])  # [bit_len*B, ...]
        assert leading_dims[0] % self.bit_slices == 0, "Leading dim must be divisible by bit_slices"
        leading_dims[0] = leading_dims[0] // self.bit_slices  # recover B
        out_shape = leading_dims + [self.out_features]
        output_concated = torch.zeros(out_shape, device=self.weight.device, dtype=x_2d.dtype)
        return output_concated

    def get_2d_weight(self, weight):
        weight_2d = weight.transpose(0, 1)
        return weight_2d

    # @profile
    def cal_x_weight_block(self, x_expanded, w_qn):
        # ===================== #
        # 权重 输入 2D 展开
        # ===================== #
        w_qn = self.get_2d_weight(w_qn)
        # ===================== #
        # 生成 2D 计算结果矩阵
        # ===================== #
        output_concat = self.gen_output_tensor(x_expanded)
        # ===================== #
        # 逐个 mapping 位置计算
        # ===================== #
        for key, weight_info in self.weight_mapping_info.items():
            start_row = weight_info['start_row']
            start_col = weight_info['start_col']
            row_num = weight_info['row_num']
            col_num = weight_info['col_num']
            # ===================== #
            # 输入数据提取
            # ===================== #
            x_split = x_expanded[:, start_row:start_row + row_num]
            # ===================== #
            # 权重提取
            # ===================== #
            w_qn_in = w_qn[start_row:start_row + row_num, start_col:start_col + col_num]
            # ===================== #
            # 逐 bit 计算
            # ===================== #
            output_bitwise = self.cal_x_bitwise(x_split, w_qn_in, start_col, col_num)
            # ===================== #
            # 输出 bit 还原
            # ===================== #
            x = bit_concat_tensor(output_bitwise, data_bit = self.input_bit, slice_bit = self.slice_bit)
            # ===================== #
            # 结果对应位置累加
            # ===================== #
            output_concat[:, start_col:start_col + col_num] += x
        return output_concat


    # @profile
    def cal_x_bitwise(self, x_expanded, w_qn, start_col, col_num):
        bit_len_batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        out_ = torch.matmul(x_expanded, w_qn)
        if self.adc_gain == self.adc_gain_min:
            init_adc_gain(self,out_, self.adc_adjust_mode)
        out_adc = self.adc_scale * out_
        out_adc = add_adc_noise(self, out_adc, start_col, col_num)
        out_adc = torch.clamp(out_adc, min = -self.adc_range - 1, max = self.adc_range)
        out_adc = round_pass(out_adc)
        return out_adc

    # @profile
    def refresh_adc_params(self):
        self.adc_range = 2 ** (self.adc_bit - 1) - 1
        self.adc_scale = get_adc_scale(self, self.adc_gain, mode = self.adc_adjust_mode)

    def forward(self, x):
        if self.use_FP:
            x = F.linear(x, self.weight, bias = self.bias)

        else:
            self.refresh_adc_params()
            # ===================== #
            # 输入量化
            # ===================== #
            x_q, in_scale = input_quant(self, x, isint = True)
            # ===================== #
            # 输入 bit 拆分
            # ===================== #
            x_expanded = bitsplit_ext.bit_split(x_q = x_q,
                                          x_bit = self.input_bit,
                                          slice_bit = self.slice_bit)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = weight_quant_noise(self, isint = True)
            # ===================== #
            # 计算
            # ===================== #
            x = self.cal_x_weight_block(x_expanded, w_qn)
            # ===================== #
            # 量化系数还原
            # ===================== #
            x = x / w_scale / in_scale / self.adc_scale
            # ===================== #
            # 输出量化
            # ===================== #
            if self.bias is not None:
                x += self.bias.view(1, -1)
            x, out_scale = output_quant(self, x, isint = False)
        return x

