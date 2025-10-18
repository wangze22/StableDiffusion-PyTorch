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
try:
    import sdk_512k.pytest.module.op_func as op
except:
    pass
from cim_toolchain_utils.utils import scatter_plt


# from memory_profiler import profile


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
class Conv2d_512k(nn.Conv2d):
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
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.input_bit = input_bit
        self.input_half_level = 2 ** input_bit / 2 - 1

        self.output_bit = output_bit
        self.output_half_level = 2 ** output_bit / 2 - 1

        self.noise_scale = noise_scale

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
        self.gen_adc_noise()

        # ===================================== #
        # 片上计算开关
        # ===================================== #
        self.on_chip = True

    def gen_adc_noise(self):
        # 增益噪声
        torch.manual_seed(self.seed)
        self.gain_noise = torch.randn(1000, device = self.weight.device) * self.gain_noise_scale
        # 偏置噪声
        self.offset_noise = torch.randn(1000, device = self.weight.device) * self.offset_noise_scale

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
        adc_gain_new = adc_gain_old
        if adc_bit_old != self.adc_bit:
            adc_range_factor = 2 ** (self.adc_bit - adc_bit_old)
            adc_gain_new *= adc_range_factor

        if dac_bit_old != self.dac_bit:
            dac_range_factor = 2 ** (self.dac_bit - dac_bit_old)
            adc_gain_new /= dac_range_factor

        if weight_bit_old != self.weight_bit:
            weight_range_factor = 2 ** (self.weight_bit - weight_bit_old)
            adc_gain_new /= weight_range_factor

        adc_gain_new = max(adc_gain_new, self.adc_gain_min * 0.8)
        self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if adc_gain_old != self.adc_gain.data.item():
            print(f'adc_gain changed: {adc_gain_old} -> {adc_gain_new}')

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
        if seed is not None:
            self.seed = seed
        self.gen_adc_noise()
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

                print(f'Initialized Step Size for Input: {self.step_size_input.data.item()}')

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
                print(f'Initialized Step Size for Weight: {self.step_size_weight.data.item()}')

            w_q, w_scale = weight_quant_lsq(data_float = self.weight,
                                            data_bit = self.weight_bit,
                                            step_size = self.step_size_weight,
                                            isint = True)

        return w_q, w_scale

    def output_quant_noise(self, x):
        x_scale = 1.0
        if self.output_quant:
            if self.step_size_output == 1:
                init_step_size = self.init_step_size(x, self.output_bit)
                self.step_size_output.data = init_step_size

                print(f'Initialized Step Size for Output: {self.step_size_output.data.item()}')
            x, x_scale = data_quant_lsq(data_float = x,
                                        data_bit = self.output_bit,
                                        step_size = self.step_size_output,
                                        isint = False)
        return x, x_scale

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
    def get_adc_scale(self):
        adc_gain = torch.clamp(self.adc_gain, min = self.adc_gain_min, max = self.adc_gain_max)
        if self.adc_adjust_mode == 'gain':
            adc_gain = round_pass(adc_gain)
        else:
            adc_range = round_pass(1 / adc_gain)
            adc_gain = 1 / adc_range
        adc_scale = adc_gain * self.adc_k
        return adc_scale

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
        row_block = 126
        for key, weight_info in self.weight_mapping_info.items():
            start_row = weight_info['start_row']
            start_col = weight_info['start_col']
            row_num = weight_info['row_num']
            col_num = weight_info['col_num']

            # 确定行范围的起始和结束
            end_row = start_row + row_num
            current_row = start_row

            # 对跨越的每个 row_block 单独进行计算
            while current_row < end_row:
                # 计算当前块的结束行，确保不会跨越 row_block 的边界
                current_block_end = min((current_row // row_block + 1) * row_block, end_row)
                block_row_num = current_block_end - current_row  # 当前块的行数

                # ===================== #
                # 输入数据提取（确保在单一 row_block 范围内）
                # ===================== #
                x_split = x_q_2d[:, current_row:current_row + block_row_num, :]

                # ===================== #
                # 权重提取（确保在单一 row_block 范围内）
                # ===================== #
                w_qn_in = w_qn[current_row:current_row + block_row_num, start_col:start_col + col_num]
                self.addr = weight_info['weight_addr']

                # ===================== #
                # 逐 bit 计算
                # ===================== #
                output_bitwise = self.cal_x_bitwise(x_split, w_qn_in, start_col, col_num)
                # ===================== #
                # 输出 bit 还原
                # ===================== #
                x = bit_concat_tensor(output_bitwise, data_bit = self.input_bit, slice_bit = self.dac_bit - 1)

                # ===================== #
                # 结果对应位置累加
                # ===================== #
                output_concat[:, :, start_col:start_col + col_num] += x

                # 移动到下一个区块的开始行
                current_row = current_block_end

        return output_concat

    def add_adc_noise(self, out_adc, start_col, col_num):
        if self.gain_noise_scale == 0 and self.offset_noise_scale == 0:
            return out_adc
        gain_noise_ = self.gain_noise[start_col:start_col + col_num]
        offset_noise_ = self.offset_noise[start_col:start_col + col_num]
        out_adc_gn = out_adc * (1 + gain_noise_)
        out_adc_off_n = out_adc_gn + self.adc_range * offset_noise_
        out_adc_off_n = (out_adc_off_n - out_adc).detach() + out_adc
        return out_adc_off_n

    # @profile
    def cal_x_bitwise(self, x_expanded, w_qn, start_col, col_num):
        bit_len_batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        x_cols = x_expanded.shape[2]
        x_expanded_512k = x_expanded.permute(0, 2, 1)
        x_expanded_512k = x_expanded_512k.reshape(-1, x_rows)
        # print(f'x_expanded.shape = {x_expanded.shape}')
        # print(f'x_expanded_512k.shape = {x_expanded_512k.shape}')
        # print(f'self.addr = {self.addr}')
        # ===================== #
        # 仿真器计算
        # ===================== #
        out_tar = torch.matmul(x_expanded.permute(0, 2, 1), w_qn)
        out_adc_tar = self.adc_scale * out_tar
        out_adc_tar = self.add_adc_noise(out_adc_tar, start_col, col_num)
        out_adc_tar = torch.clamp(out_adc_tar, min = -self.adc_range - 1, max = self.adc_range)
        out_adc_tar = round_pass(out_adc_tar)
        # ===================== #
        if self.on_chip:
            # print(f'self.addr = {self.addr}')
            _, out_ = op.cima512k_batch_calc(self.addr, x_expanded_512k.detach().numpy(),
                                             n_bias = 2,
                                             idac_fs = self.adc_gain)
            out_ = out_.reshape(bit_len_batch, x_cols, -1)
            out_ = torch.tensor(out_).to(x_expanded_512k.device)
            # print(f'out_.shape = {out_.shape}')
            scatter_plt(out_, out_adc_tar)
            out_adc = (out_ - out_adc_tar).detach() + out_adc_tar
        else:
            out_adc = out_adc_tar
        if self.adc_gain == self.adc_gain_min:
            self.init_adc_gain(out_tar)
        return out_adc

    def init_adc_gain(self, out_):
        if out_.abs().max() != 0:
            adc_scale_ideal = self.adc_range / out_.abs().max()
            adc_gain_ideal = adc_scale_ideal / self.adc_k
            adc_gain_ideal = max(adc_gain_ideal, 0.8 * self.adc_gain_min)
            self.adc_gain.data = torch.tensor(adc_gain_ideal, device = self.adc_gain.device)
            self.adc_scale = self.get_adc_scale()
            print(f'Initialized Adc Gain: {self.adc_gain.data}')

    # @profile
    def refresh_adc_params(self):
        self.adc_range = 2 ** (self.adc_bit - 1) - 1
        self.adc_scale = self.get_adc_scale()

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
            x_q, in_scale = self.input_quant_noise(x)
            # ===================== #
            # 输入 bit 拆分
            # ===================== #
            x_expanded = bit_split_tensor(x_q = x_q,
                                          x_bit = self.input_bit,
                                          slice_bit = self.slice_bit)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            w_qn = add_noise(w_qn, n_scale = self.noise_scale)
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
            x, out_scale = self.output_quant_noise(x)
        return x


class Linear_512k(nn.Linear):
    def __init__(self, in_features, out_features,
                 weight_bit,
                 input_bit,
                 output_bit,
                 noise_scale,
                 dac_bit,
                 adc_bit,
                 adc_k,
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
        self.weight_half_level = 2 ** weight_bit / 2 - 1
        self.input_bit = input_bit
        self.input_half_level = 2 ** input_bit / 2 - 1

        self.output_bit = output_bit
        self.output_half_level = 2 ** output_bit / 2 - 1

        self.noise_scale = noise_scale

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
        self.gen_adc_noise()
        # ===================================== #
        # 片上计算开关
        # ===================================== #
        self.on_chip = True

    def gen_adc_noise(self):
        # 增益噪声
        torch.manual_seed(self.seed)
        self.gain_noise = torch.randn(1000, device = self.weight.device) * self.gain_noise_scale
        # 偏置噪声
        self.offset_noise = torch.randn(1000, device = self.weight.device) * self.offset_noise_scale

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
        adc_gain_new = adc_gain_old
        if adc_bit_old != self.adc_bit:
            adc_range_factor = 2 ** (self.adc_bit - adc_bit_old)
            adc_gain_new *= adc_range_factor

        if dac_bit_old != self.dac_bit:
            dac_range_factor = 2 ** (self.dac_bit - dac_bit_old)
            adc_gain_new /= dac_range_factor

        if weight_bit_old != self.weight_bit:
            weight_range_factor = 2 ** (self.weight_bit - weight_bit_old)
            adc_gain_new /= weight_range_factor

        adc_gain_new = max(adc_gain_new, 0.8 * self.adc_gain_min)
        self.adc_gain.data = torch.tensor(adc_gain_new, device = self.adc_gain.device)

        if adc_gain_old != self.adc_gain.data.item():
            print(f'adc_gain changed: {adc_gain_old} -> {adc_gain_new}')

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
        if seed is not None:
            self.seed = seed
        self.gen_adc_noise()
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

                print(f'Initialized Step Size for Input: {self.step_size_input.data.item()}')

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
                print(f'Initialized Step Size for Weight: {self.step_size_weight.data.item()}')

            w_q, w_scale = weight_quant_lsq(data_float = self.weight,
                                            data_bit = self.weight_bit,
                                            step_size = self.step_size_weight,
                                            isint = True)

        return w_q, w_scale

    def output_quant_noise(self, x):
        x_scale = 1.0
        if self.output_quant:
            if self.step_size_output == 1:
                init_step_size = self.init_step_size(x, self.output_bit)
                self.step_size_output.data = init_step_size

                print(f'Initialized Step Size for Output: {self.step_size_output.data.item()}')
            x, x_scale = data_quant_lsq(data_float = x,
                                        data_bit = self.output_bit,
                                        step_size = self.step_size_output,
                                        isint = False)
        return x, x_scale

    def get_adc_scale(self):
        adc_gain = torch.clamp(self.adc_gain, min = self.adc_gain_min, max = self.adc_gain_max)
        if self.adc_adjust_mode == 'gain':
            adc_gain = round_pass(adc_gain)
        else:
            adc_range = round_pass(1 / adc_gain)
            adc_gain = 1 / adc_range
        adc_scale = adc_gain * self.adc_k
        return adc_scale

    # @profile
    def gen_output_tensor(self, x_2d):
        batch_num = x_2d.shape[0] // self.bit_slices
        output_concated = torch.zeros([batch_num,
                                       self.out_features],
                                      device = self.weight.device)
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
        row_block = 126
        for key, weight_info in self.weight_mapping_info.items():
            start_row = weight_info['start_row']
            start_col = weight_info['start_col']
            row_num = weight_info['row_num']
            col_num = weight_info['col_num']

            # 确定行范围的起始和结束
            end_row = start_row + row_num
            current_row = start_row

            # 对跨越的每个 row_block 单独进行计算
            while current_row < end_row:
                # 计算当前块的结束行，确保不会跨越 row_block 的边界
                current_block_end = min((current_row // row_block + 1) * row_block, end_row)
                block_row_num = current_block_end - current_row  # 当前块的行数

                # ===================== #
                # 输入数据提取（确保在单一 row_block 范围内）
                # ===================== #
                x_split = x_expanded[:, current_row:current_row + block_row_num, :]

                # ===================== #
                # 权重提取（确保在单一 row_block 范围内）
                # ===================== #
                w_qn_in = w_qn[current_row:current_row + block_row_num, start_col:start_col + col_num]
                self.addr = weight_info['weight_addr']

                # ===================== #
                # 逐 bit 计算
                # ===================== #
                output_bitwise = self.cal_x_bitwise(x_split, w_qn_in, start_col, col_num)

                # ===================== #
                # 输出 bit 还原
                # ===================== #
                x = bit_concat_tensor(output_bitwise, data_bit = self.input_bit, slice_bit = self.dac_bit - 1)

                # ===================== #
                # 结果对应位置累加
                # ===================== #
                output_concat[:, :, start_col:start_col + col_num] += x

                # 移动到下一个区块的开始行
                current_row = current_block_end
        return output_concat

    def add_adc_noise(self, out_adc, start_col, col_num):
        if self.gain_noise_scale == 0 and self.offset_noise_scale == 0:
            return out_adc
        gain_noise_ = self.gain_noise[start_col:start_col + col_num]
        offset_noise_ = self.offset_noise[start_col:start_col + col_num]
        out_adc_gn = out_adc * (1 + gain_noise_)
        out_adc_off_n = out_adc_gn + self.adc_range * offset_noise_
        out_adc_off_n = (out_adc_off_n - out_adc).detach() + out_adc
        return out_adc_off_n

    # @profile
    def cal_x_bitwise(self, x_expanded, w_qn, start_col, col_num):
        bit_len_batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        x_cols = x_expanded.shape[2]
        # ===================== #
        # 仿真器计算
        # ===================== #
        out_tar = torch.matmul(x_expanded, w_qn)
        out_adc_tar = self.adc_scale * out_tar
        out_adc_tar = self.add_adc_noise(out_adc_tar, start_col, col_num)
        out_adc_tar = torch.clamp(out_adc_tar, min = -self.adc_range - 1, max = self.adc_range)
        out_adc_tar = round_pass(out_adc_tar)
        # ===================== #
        if self.on_chip:
            x_expanded_512k = x_expanded.reshape(-1, x_rows)
            _, out_ = op.cima512k_batch_calc(self.addr, x_expanded_512k.detach().numpy(),
                                             n_bias = 2,
                                             idac_fs = self.adc_gain)
            out_ = out_.reshape(bit_len_batch, x_cols, -1)
            out_ = torch.tensor(out_).to(x_expanded.device)
            out_adc = (out_ - out_adc_tar).detach() + out_adc_tar
        else:
            out_adc = out_adc_tar
        if self.adc_gain == self.adc_gain_min:
            self.init_adc_gain(out_tar)
        return out_adc

    def init_adc_gain(self, out_):
        if out_.abs().max() != 0:
            adc_scale_ideal = self.adc_range / out_.abs().max()
            adc_gain_ideal = adc_scale_ideal / self.adc_k
            adc_gain_ideal = max(adc_gain_ideal, 0.8 * self.adc_gain_min)
            self.adc_gain.data = torch.tensor(adc_gain_ideal, device = self.adc_gain.device)
            self.adc_scale = self.get_adc_scale()
            print(f'Initialized Adc Gain: {self.adc_gain.data}')

    # @profile
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
            x_q, in_scale = self.input_quant_noise(x)
            # ===================== #
            # 输入 bit 拆分
            # ===================== #
            x_expanded = bit_split_tensor(x_q = x_q,
                                          x_bit = self.input_bit,
                                          slice_bit = self.slice_bit)
            # ===================== #
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            w_qn = add_noise(w_qn, n_scale = self.noise_scale)
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
            x, out_scale = self.output_quant_noise(x)
        return x


def bit_split_tensor(x_q, x_bit, slice_bit):
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
    bit_data_list = torch.cat(bit_data_list, 0)
    return bit_data_list


def bit_concat_tensor(bitwise_data, data_bit, slice_bit):
    bit_len = int(math.ceil((data_bit - 1) / slice_bit))
    bitwise_data = torch.chunk(bitwise_data, chunks = bit_len, dim = 0)
    data_sum = torch.zeros_like(bitwise_data[0], device = bitwise_data[0].device)
    for i, bit_data in enumerate(bitwise_data):
        data_sum += bit_data * 2 ** (i * slice_bit)
    return data_sum
