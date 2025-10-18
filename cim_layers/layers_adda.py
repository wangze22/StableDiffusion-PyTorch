import os

import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cim_layers.quant_noise_utils import *
from cim_runtime.cim_expansion_LUT import generate_lookup_table
from cim_runtime.cim_expansion_LUT import lookup_tables

# from
# 从环境变量获取期望的CUDA device
desired_device = os.getenv('CUDA_DEVICE', 'cuda:0')

# 检查CUDA是否可用，以及指定的CUDA device是否存在
if torch.cuda.is_available():
    device = torch.device(desired_device)
else:
    device = torch.device('cpu')


def data_quant(data_float, data_bit, isint = False):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    # reg_shift_mode -> force half_level to be exponent of 2, i.e., half_level = 2^n (n is integer)
    assert data_bit >= 2
    half_level = 2 ** (data_bit - 1) - 1

    data_range = abs(data_float).max()

    data_quantized = (data_float / data_range * half_level).round()
    quant_scale = 1 / data_range * half_level

    if not isint:
        data_quantized = data_quantized / half_level * data_range
        quant_scale = 1.0

    return data_quantized, quant_scale


def add_noise(weight, n_scale = 0.074):
    w_range = weight.max() - weight.min()
    w_noise = w_range * n_scale * torch.randn_like(weight)
    weight_noise = weight + w_noise
    return weight_noise


def input_expansion(input_matrix, dac_bits = 2, input_bits = 8):
    if isinstance(input_matrix, np.ndarray):
        input_matrix = torch.from_numpy(input_matrix)

    if not torch.any(input_matrix):
        return input_matrix, 1

    if (dac_bits, input_bits) not in lookup_tables:
        lookup_table = generate_lookup_table(dac_bits = dac_bits, input_bits = input_bits)
    else:
        lookup_table = lookup_tables[(dac_bits, input_bits)]

    value_range = 2 ** (input_bits - 1) - 1
    idx = (input_matrix + value_range).to(torch.long)
    input_expanded = lookup_table[idx]

    # 获取张量的维度
    dims = input_expanded.dim()

    # 创建一个新的维度顺序列表，其中第一个维度挪到最后一个，其他的维度向前挪一个
    new_order = (torch.arange(dims) - 1) % dims

    # 使用 permute 重新排列维度
    input_expanded = input_expanded.permute(*new_order.tolist())

    return input_expanded


class Conv2dCimFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_int, weight_int, stride, padding, dilation, groups,
                feature_bit, dac_bit, adc_bit):
        ctx.save_for_backward(input_int, weight_int)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.dac_bit = dac_bit
        input_expanded = input_expansion(input_int, dac_bits = dac_bit, input_bits = feature_bit)
        ctx.input_expanded = input_expanded
        e, b, c, h, w = input_expanded.shape
        # 确保输入维度适合 conv2d
        input_expanded = input_expanded.reshape(-1, c, h, w)

        # 进行卷积运算
        output_tensor_expanded = F.conv2d(input_expanded, weight_int,
                                          stride = stride, padding = padding,
                                          dilation = dilation, groups = groups)
        _, c, h, w = output_tensor_expanded.shape
        output_tensor_expanded = output_tensor_expanded.reshape(e, b, c, h, w)
        return output_tensor_expanded

    @staticmethod
    def backward(ctx, grad_output):
        # -------------------------- #
        # 打印梯度在不同bit位的均值
        # grad_output 梯度，低比特位在前，高比特位在后，e维度越大，梯度越大
        # reshaped_g = grad_output.view(grad_output.shape[0], -1)
        # m = reshaped_g.mean(dim = 1)
        # print(m)
        # -------------------------- #
        input_expanded = ctx.input_expanded
        e, b, c, h, w = input_expanded.shape
        # ======================= #
        # grad_output 梯度合并
        # ======================= #
        # -------------------------- #
        # 方法1：对高低比特进行加权计算
        # -------------------------- #
        dac_bit = ctx.dac_bit
        coefficient = torch.arange(0, e,
                                   device = grad_output.device) * (dac_bit - 1)
        coefficient = 2.0 ** (-coefficient)
        coefficient = coefficient.view(-1, 1, 1, 1, 1)
        grad_output = (grad_output * coefficient).sum(dim = 0)
        # -------------------------- #
        # 方法2：只保留最高bit位的梯度
        # -------------------------- #
        # grad_output = grad_output[e - 1]
        # -------------------------- #
        # 方法3：直接求sum
        # -------------------------- #
        # grad_output = grad_output.sum(dim = 0)
        # -------------------------- #
        input_int, weight_int = ctx.saved_tensors
        grad_input = grad_weight = None
        # -------------------------- #
        # input_expanded = ctx.input_expanded
        # e, b, c, h, w = input_expanded.shape
        # input_expanded_reshape = input_expanded.reshape(-1, c, h, w)
        # grad_input = grad_weight =  None
        # dac_bit = ctx.dac_bit
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input_int.shape, weight_int, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input_int, weight_int.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups)

        return grad_input, grad_weight, None, None, None, None, None, None, None


class Conv2d_ADDA_aware(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 weight_bit,
                 feature_bit,
                 clamp_std,
                 noise_scale,
                 adc_bit,
                 dac_bit,
                 offset_noise_scale = None,
                 gain_noise_scale = None,
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
        self.feature_bit = feature_bit
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.adc_bit = adc_bit
        self.dac_bit = dac_bit
        self.offset_noise_scale = offset_noise_scale
        self.gain_noise_scale = gain_noise_scale
        # adc_scale 对应 (ADC计算增益*DAC计算增益)/(ADC写校验增益*DAC写校验增益)
        if isinstance(kernel_size, tuple):
            kernel_area = kernel_size[0] * kernel_size[1]
        else:
            kernel_area = kernel_size ** 2
        self.kernel_area = kernel_area
        self.adc_half_level = 2 ** (self.adc_bit - 1) - 1
        # self.adc_scale = nn.Parameter(torch.tensor(10.0))
        self.init_adc_scale()

    def apply_ADC_scale_clamp(self, data_in):
        data_adc_scaled = data_in * abs(self.adc_scale)
        data_adc_clamped = torch.clamp(data_adc_scaled, -self.adc_half_level - 1, self.adc_half_level)
        # data_adc_clamped = data_adc_scaled
        return data_adc_clamped

    def init_adc_scale(self):
        rows_size = self.in_channels * self.kernel_area
        adc_weight_scale = 2 ** (self.adc_bit - self.weight_bit)
        adc_dac_scale = 2 ** (self.dac_bit - 1)
        adc_scale = 1 / rows_size / adc_dac_scale * adc_weight_scale
        self.adc_scale = nn.Parameter(torch.tensor(adc_scale))

    def update_para(self,
                    weight_bit = None,
                    feature_bit = None,
                    clamp_std = None,
                    noise_scale = None,
                    dac_bit = None,
                    adc_bit = None,
                    gain_noise_scale = None,
                    offset_noise_scale = None,
                    ):
        if weight_bit is not None:
            self.weight_bit = weight_bit
        if feature_bit is not None:
            self.feature_bit = feature_bit
        if clamp_std is not None:
            self.clamp_std = clamp_std
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if dac_bit is not None:
            self.dac_bit = dac_bit
        if adc_bit is not None:
            self.adc_bit = adc_bit
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale
        self.init_adc_scale()

    def use_FP(self):
        return (self.weight_bit <= 0 and
                self.feature_bit <= 0 and
                self.adc_bit <= 0 and
                self.dac_bit <= 0 and
                self.clamp_std <= 0 and
                self.noise_scale <= 0 and
                self.gain_noise_scale <= 0 and
                self.offset_noise_scale <= 0)

    def forward(self, x):
        if self.use_FP():
            output = self._conv_forward(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            x_q, x_scale = DataQuant.apply(x, self.feature_bit, True)
            # ===================== #
            # 权重截断&量化
            # ===================== #
            # w_q 对应权重的写校验值
            w_std = self.weight.std()
            if self.clamp_std > 0:
                w_c = torch.clamp(self.weight, -w_std * self.clamp_std, w_std * self.clamp_std)
            else:
                w_c = self.weight + 0
            w_q, w_scale = DataQuant.apply(w_c, self.weight_bit, True)
            w_qn = add_noise(w_q, n_scale = self.noise_scale)
            # ===================== #
            # 高低位展开&卷积
            # ===================== #
            # output = F.conv2d(x_q, w_qn, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # output /= (w_scale * x_scale)
            output_tensor_expanded = Conv2dCimFunction.apply(x_q, w_qn,
                                                             self.stride, self.padding, self.dilation, self.groups,
                                                             self.feature_bit, self.dac_bit, self.adc_bit
                                                             )

            # ===================== #
            # ADC 截断
            # ===================== #
            output_tensor_expanded = self.apply_ADC_scale_clamp(output_tensor_expanded)
            # ===================== #
            # 移位求和
            # ===================== #
            coefficient = torch.arange(0, output_tensor_expanded.shape[0],
                                       device = output_tensor_expanded.device) * (self.dac_bit - 1)
            coefficient = 2 ** coefficient
            coefficient = coefficient.view(-1, 1, 1, 1, 1)

            # 对结果乘以系数并求和
            output = (output_tensor_expanded * coefficient).sum(dim = 0)
            output, _ = DataQuant.apply(output, self.feature_bit, False)
            output = output / (w_scale * x_scale * abs(self.adc_scale))
            output += self.bias.view(1, -1, 1, 1)
        return output


class LinearCimFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_int, weight_int,
                feature_bit, dac_bit, adc_bit):
        ctx.save_for_backward(input_int, weight_int)
        ctx.dac_bit = dac_bit
        input_expanded = input_expansion(input_int, dac_bits = dac_bit, input_bits = feature_bit)
        ctx.input_expanded = input_expanded
        e, b, s = input_expanded.shape
        # 确保输入维度适合 conv2d
        input_expanded = input_expanded.reshape(-1, s)

        # 进行卷积运算
        output_tensor_expanded = F.linear(input_expanded, weight_int)
        _, s = output_tensor_expanded.shape
        output_tensor_expanded = output_tensor_expanded.reshape(e, b, s)
        return output_tensor_expanded

    @staticmethod
    def backward(ctx, grad_output):
        # -------------------------- #
        # 打印梯度在不同bit位的均值
        # reshaped_g = grad_output.view(grad_output.shape[0], -1)
        # m = reshaped_g.mean(dim = 1)
        # print(m)
        # -------------------------- #
        input_expanded = ctx.input_expanded
        e, b, s = input_expanded.shape
        # ======================= #
        # grad_output 梯度合并
        # ======================= #
        # -------------------------- #
        # 方法1：对高低比特进行加权计算
        # -------------------------- #
        dac_bit = ctx.dac_bit
        coefficient = torch.arange(0, e,
                                   device = grad_output.device) * (dac_bit - 1)
        coefficient = 2.0 ** (-coefficient)
        coefficient = coefficient.view(-1, 1, 1)
        grad_output = (grad_output * coefficient).sum(dim = 0)
        # -------------------------- #
        # 方法2：只保留最高bit位的梯度
        # -------------------------- #
        # grad_output = grad_output[e - 1]
        # -------------------------- #
        # 方法3：直接求sum
        # -------------------------- #
        # grad_output = grad_output.sum(dim = 0)
        # -------------------------- #
        input_int, weight_int = ctx.saved_tensors
        grad_input = grad_weight = None
        # input_expanded = ctx.input_expanded
        # e, b, c, h, w = input_expanded.shape
        # input_expanded_reshape = input_expanded.reshape(-1, c, h, w)
        # grad_input = grad_weight =  None
        # dac_bit = ctx.dac_bit
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_int)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input_int)

        return grad_input, grad_weight, None, None, None


class Linear_ADDA_aware(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_bit,
                 feature_bit,
                 clamp_std,
                 noise_scale,
                 adc_bit,
                 dac_bit,
                 bias = True,
                 offset_noise_scale = None,
                 gain_noise_scale = None,
                 ):
        super().__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.feature_bit = feature_bit
        self.clamp_std = clamp_std
        self.noise_scale = noise_scale
        self.adc_bit = adc_bit
        self.dac_bit = dac_bit
        self.offset_noise_scale = offset_noise_scale
        self.gain_noise_scale = gain_noise_scale
        # adc_scale 对应 (ADC计算增益*DAC计算增益)/(ADC写校验增益*DAC写校验增益)
        self.init_adc_scale()
        self.adc_half_level = 2 ** (self.adc_bit - 1) - 1
        # self.adc_scale = nn.Parameter(torch.tensor(10.0))

    def init_adc_scale(self):
        rows_size = self.in_features
        adc_weight_scale = 2 ** (self.adc_bit - self.weight_bit)
        adc_dac_scale = 2 ** (self.dac_bit - 1)
        adc_scale = 1 / rows_size / adc_dac_scale * adc_weight_scale
        self.adc_scale = nn.Parameter(torch.tensor(adc_scale))

    def apply_ADC_scale_clamp(self, data_in):
        data_adc_scaled = data_in * abs(self.adc_scale)
        data_adc_clamped = torch.clamp(data_adc_scaled, -self.adc_half_level - 1, self.adc_half_level)
        return data_adc_clamped

    def update_para(self,
                    weight_bit = None,
                    feature_bit = None,
                    clamp_std = None,
                    noise_scale = None,
                    dac_bit = None,
                    adc_bit = None,
                    gain_noise_scale = None,
                    offset_noise_scale = None,
                    ):
        if weight_bit is not None:
            self.weight_bit = weight_bit
        if feature_bit is not None:
            self.feature_bit = feature_bit
        if clamp_std is not None:
            self.clamp_std = clamp_std
        if noise_scale is not None:
            self.noise_scale = noise_scale
        if dac_bit is not None:
            self.dac_bit = dac_bit
        if adc_bit is not None:
            self.adc_bit = adc_bit
        if gain_noise_scale is not None:
            self.gain_noise_scale = gain_noise_scale
        if offset_noise_scale is not None:
            self.offset_noise_scale = offset_noise_scale
        self.init_adc_scale()

    def use_FP(self):
        return (self.weight_bit <= 0 and
                self.feature_bit <= 0 and
                self.adc_bit <= 0 and
                self.dac_bit <= 0 and
                self.clamp_std <= 0 and
                self.noise_scale <= 0 and
                self.gain_noise_scale <= 0 and
                self.offset_noise_scale <= 0)

    def forward(self, x):
        if self.use_FP():
            output = F.linear(x, self.weight, bias = self.bias)

        else:
            # ===================== #
            # 输入量化
            # ===================== #
            x_q, x_scale = DataQuant.apply(x, self.feature_bit, True)
            # ===================== #
            # 权重截断&量化
            # ===================== #
            # w_q 对应权重的写校验值
            w_std = self.weight.std()
            if self.clamp_std > 0:
                w_c = torch.clamp(self.weight, -w_std * self.clamp_std, w_std * self.clamp_std)
            else:
                w_c = self.weight + 0
            w_q, w_scale = DataQuant.apply(w_c, self.weight_bit, True)
            w_qn = add_noise(w_q, n_scale = self.noise_scale)
            # ===================== #
            # 高低位展开&卷积
            # ===================== #
            # output = F.conv2d(x_q, w_qn, self.bias, self.stride, self.padding, self.dilation, self.groups)
            # output /= (w_scale * x_scale)
            output_tensor_expanded = LinearCimFunction.apply(x_q, w_qn,
                                                             self.feature_bit, self.dac_bit, self.adc_bit
                                                             )

            # ===================== #
            # ADC 截断
            # ===================== #
            output_tensor_expanded = self.apply_ADC_scale_clamp(output_tensor_expanded)
            # ===================== #
            # 移位求和
            # ===================== #
            coefficient = torch.arange(0, output_tensor_expanded.shape[0],
                                       device = output_tensor_expanded.device) * (self.dac_bit - 1)
            coefficient = 2 ** coefficient
            coefficient = coefficient.view(-1, 1, 1)

            # 对结果乘以系数并求和
            output = (output_tensor_expanded * coefficient).sum(dim = 0)
            output, _ = DataQuant.apply(output, self.feature_bit, False)
            output = output / (w_scale * x_scale * abs(self.adc_scale))
            output += self.bias.view(1, -1)
        return output


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    conv_cim = Conv2d_ADDA_aware(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 0,
                                 groups = 1, bias = False,
                                 weight_bit = 4, feature_bit = 8, noise_scale = 0.0,
                                 adc_bit = 4, dac_bit = 2, clamp_std = 0).to('cuda')
    conv = nn.Conv2d(in_channels = 32, out_channels = 128, kernel_size = 3, stride = 1, padding = 0,
                     groups = 1, bias = False).to('cuda')
    conv.weight.data = conv_cim.weight.data + 0

    opt = optim.Adam(conv_cim.parameters(), lr = 0.001)
    opt2 = optim.Adam([conv_cim.adc_scale], lr = 0.1)
    print(conv_cim.adc_scale)
    loss_list = []
    adc_scale_list = []
    weight_diff_list = []
    for i in range(500):
        x = torch.randn(32, 32, 64, 64, device = 'cuda')
        y_cim = conv_cim(x)
        y = conv(x)
        loss = torch.sum((y_cim - y) ** 2)
        opt.zero_grad()
        opt2.zero_grad()
        loss.backward()
        if i % 99 == 0:
            print(f'loss = {loss}')
        adc_scale_list.append(conv_cim.adc_scale.data.cpu())
        weight_diff = ((conv_cim.weight.data - conv.weight.data) ** 2).sum()
        weight_diff_list.append(weight_diff.item())
        # print(f'y = {y}')
        # print(f'y_cim = {y_cim}')
        # print(f'conv.grad = {conv.weight.grad}')
        # print(f'conv_cim.grad = {conv_cim.weight.grad}')
        # print(f'conv_cim.adc_scale.grad = {conv_cim.adc_scale.grad}')
        # opt.step()
        opt2.step()
        loss_list.append(loss.item())
    print(conv_cim.adc_scale)
    plt.plot(loss_list)
    plt.title(f'Loss')
    plt.yscale('log')
    plt.show()

    plt.plot(adc_scale_list)
    plt.title(f'ADC Scale')
    plt.show()

    plt.plot(weight_diff_list)
    plt.title(f'Weight Difference')
    plt.show()
