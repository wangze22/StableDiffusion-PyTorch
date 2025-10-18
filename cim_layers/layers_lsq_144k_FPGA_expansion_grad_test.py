# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

import cim_layers.layers_qn_lsq_adda_cim as adda_cim

from cim_layers.quant_noise_utils import *
from cim_toolchain_utils.utils import *


# from memory_profiler import profile


# ====================================================================== #
# Customized nn.Module layers for quantization and noise adding
# ====================================================================== #
class Conv2d_lsq_144k(adda_cim.Conv2d_lsq_adda_cim):
    def weight_quant_noise(self):
        w_qn = self.weight
        w_scale = 1.0
        if self.weight_quant:
            if self.step_size_weight == 1:
                init_step_size = self.init_step_size(self.weight, self.weight_bit)
                self.step_size_weight.data = init_step_size
                print(f'Initialized Step Size for Weight: {self.step_size_weight}')

            w_qn, w_scale = weight_quant_lsq(data_float = self.weight,
                                             data_bit = self.weight_bit,
                                             step_size = self.step_size_weight,
                                             isint = True)
        return w_qn, w_scale

    def gen_output_tensor(self, x_2d):
        batch_num = x_2d.shape[0]
        output_rows = x_2d.shape[2]
        output_concated = torch.zeros([batch_num,
                                       output_rows,
                                       self.out_channels],
                                      device = self.weight.device)
        return output_concated

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
            array_idx = weight_info['array_idx']
            self.weight_addr = weight_info['weight_addr']
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
            x = self.cal_x_bitwise(x_split, w_qn_in)
            # ===================== #
            # 结果对应位置累加
            # ===================== #
            output_concat[:, :, start_col:start_col + col_num] += x
        return output_concat

    # @profile
    def cal_x_bitwise(self, x_expanded, w_qn):
        batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        x_cols = x_expanded.shape[2]
        x_expanded = x_expanded.permute(0, 2, 1)
        # ===================== #
        # 144k片上计算
        # ===================== #
        x_expanded_144k = x_expanded.reshape(-1, x_rows)
        x_expanded_pos = x_expanded_144k + 0
        x_expanded_neg = x_expanded_144k + 0
        x_expanded_pos[x_expanded_pos < 0] = 0
        x_expanded_neg[x_expanded_neg > 0] = 0

        out_pos = torch.matmul(x_expanded_pos, w_qn) * self.adc_scale
        out_neg = torch.matmul(x_expanded_neg, w_qn) * self.adc_scale

        out_array = out_pos + out_neg
        out_array = torch.clamp(out_array, -127, 127)
        out_array = torch.round(out_array)
        out_array = out_array.reshape(batch, x_cols, -1)
        return out_array

    def forward(self, x):
        if self.use_FP():
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
            # 权重截断&量化 + 噪声
            # ===================== #
            w_qn, w_scale = self.weight_quant_noise()
            # ===================== #
            # 144k 计算
            # ===================== #
            output_concated = self.cal_x_weight_block(x_q, w_qn)
            x = self.fold_output(output_concated)
            # ===================== #
            # torch 计算
            # ===================== #
            x_tar = self._conv_forward(x_q, w_qn, bias = None)
            # ===================== #
            # 量化系数还原
            # ===================== #
            x = x / w_scale / in_scale / self.adc_scale
            x_tar = x_tar / w_scale / in_scale
            # scatter_plt(x, x_tar, title = f'{self.name}', save_fig = 1, show_fig = 0)
            # ===================== #
            # 求导传递
            # ===================== #
            x = (x - x_tar).detach() + x_tar
            # ===================== #
            # 输出量化
            # ===================== #
            if self.bias is not None:
                x += self.bias.view(1, -1, 1, 1)
            x, out_scale = self.output_quant_noise(x)
        return x


class Linear_lsq_144k(adda_cim.Linear_lsq_adda_cim):
    def weight_quant_noise(self):
        w_qn = self.weight
        w_scale = 1.0
        if self.weight_quant:
            if self.step_size_weight == 1:
                init_step_size = self.init_step_size(self.weight, self.weight_bit)
                self.step_size_weight.data = init_step_size
                print(f'Initialized Step Size for Weight: {self.step_size_weight}')

            w_qn, w_scale = weight_quant_lsq(data_float = self.weight,
                                             data_bit = self.weight_bit,
                                             step_size = self.step_size_weight,
                                             isint = True)
        return w_qn, w_scale

    def cal_x_weight_block(self, x_expanded, w_qn):
        # ===================== #
        # 生成 2D 计算结果矩阵
        # ===================== #
        output_concat = self.gen_output_tensor(x_expanded)
        # ===================== #
        # 逐个 mapping 位置计算
        # ===================== #
        for key, weight_info in self.weight_mapping_info.items():
            array_idx = weight_info['array_idx']
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
            output_bitwise = self.cal_x_bitwise(x_split, w_qn_in)
            # ===================== #
            # 输出 bit 还原
            # ===================== #
            x = adda_cim.bit_concat_tensor(output_bitwise, data_bit = self.input_bit, slice_bit = self.slice_bit)
            # ===================== #
            # 结果对应位置累加
            # ===================== #
            output_concat[:, start_col:start_col + col_num] += x
        return output_concat

    # @profile
    def cal_x_bitwise(self, x_expanded, w_qn):
        bit_len_batch = x_expanded.shape[0]
        x_rows = x_expanded.shape[1]
        x_cols = x_expanded.shape[2]
        # ===================== #
        # 为了进行梯度传递
        # ===================== #
        out_ = torch.matmul(x_expanded, w_qn)
        out_adc = self.adc_scale * out_
        out_adc = torch.clamp(out_adc, min = -self.adc_range - 1, max = self.adc_range)
        out_adc_grad = round_pass(out_adc)
        # ===================== #
        # 144k片上计算
        # ===================== #
        x_expanded_pos = x_expanded + 0
        x_expanded_neg = x_expanded + 0
        x_expanded_pos[x_expanded_pos < 0] = 0
        x_expanded_neg[x_expanded_neg > 0] = 0
        out_pos = torch.matmul(x_expanded_pos, w_qn) * self.adc_scale
        out_neg = np.matmul(x_expanded_neg, to_numpy(w_qn))
        out_array = out_pos + out_neg
        out_array = to_tensor(out_array, device = w_qn.device)
        # ===================== #
        # 最终输出
        # ===================== #
        out = (out_array - out_adc_grad).detach() + out_adc_grad
        return out
