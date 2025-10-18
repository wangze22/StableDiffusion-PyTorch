import torch
import math

# ================================== #
# Quantization and noise adding
# ================================== #
# Quantize input data
def data_quant(data_float, data_bit, isint = False):
    if data_bit == 0:
        return data_float, 1.0

    assert data_bit >= 2

    half_level = 2 ** (data_bit - 1) - 1

    data_range = abs(data_float).max()
    if data_range == 0:
        return data_float, 1.0
    data_quantized = (data_float / data_range * half_level).round()
    quant_scale = 1 / data_range * half_level

    if not isint:
        data_quantized = data_quantized / half_level * data_range
        quant_scale = 1.0

    # if torch.isnan(data_quantized).any():
    #     raise RuntimeError(f'NaN in data_quantized: {data_quantized}')
    return data_quantized, quant_scale


def data_quant_pass(data_float, data_bit, isint = False):
    if data_bit == 0:
        return data_float, 1

    assert data_bit >= 2

    half_level = 2 ** (data_bit - 1) - 1
    data_range = data_float.detach().abs().max()

    quant_scale = 1 / data_range * half_level
    data_scaled = data_float * quant_scale
    data_quantized = round_pass(data_scaled)

    if not isint:
        data_quantized = data_quantized / half_level * data_range
        quant_scale = 1.0

    return data_quantized, quant_scale


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def grad_scale_factor(data_range, x):
    return 1.0 / ((data_range * x.numel()) ** 0.5)


def clamp_pass(x, min, max):
    y = torch.clamp(x, min, max)
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = torch.round(x)
    y_grad = x
    return (y - y_grad).detach() + y_grad

def round_pass_exp(x):
    shift_factor = torch.log2(torch.tensor(1.5)) - 0.5
    y = torch.round(x - shift_factor)
    y_grad = x
    return (y - y_grad).detach() + y_grad


def floor_pass(x):
    x_abs = x.abs().detach()
    x_sign = x.sign().detach()
    y = torch.floor(x_abs) * x_sign
    y_grad = x
    return (y - y_grad).detach() + y_grad


def floor_no_pass(x):
    y = torch.where(x >= 0, x.floor(), x.ceil())  # 正数向下取整，负数向上取整
    return y


# Add noise to input data
def add_noise(weight, n_scale = 0.074):
    if n_scale == 0:
        return weight
    w_range = weight.max() - weight.min()
    w_noise = w_range * n_scale * torch.randn_like(weight)
    weight_noise = weight + w_noise
    return weight_noise


def data_quant_lsq(data_float, data_bit, step_size, isint = False):
    assert data_bit > 0
    quant_scale = (1 / step_size).detach()

    data_range = 2 ** (data_bit - 1) - 1
    gradScaleFactor = grad_scale_factor(data_range, data_float)

    step_size = grad_scale(step_size, gradScaleFactor)

    data_scaled = data_float / step_size
    data_clamped = torch.clamp(data_scaled, -data_range, data_range)

    data_quantized = round_pass(data_clamped)

    if not isint:
        data_quantized = data_quantized * step_size
        quant_scale = 1.0
    if isint:
        data_quantized = data_quantized * step_size / step_size.detach()
    return data_quantized, quant_scale

def weight_quant_floor(data_float, data_bit, step_size, isint = False):
    assert data_bit > 0
    quant_scale = (1 / step_size).detach()

    data_range = 2 ** (data_bit - 1) - 1

    data_scaled = data_float / step_size
    data_clamped = torch.clamp(data_scaled, -data_range, data_range)

    data_quantized = floor_pass(data_clamped)

    if not isint:
        data_quantized = data_quantized * step_size
        quant_scale = 1.0
    if isint:
        data_quantized = data_quantized * step_size / step_size.detach()
    return data_quantized, quant_scale

def weight_quant_round(data_float, data_bit, step_size, isint = False):
    assert data_bit > 0
    quant_scale = (1 / step_size).detach()

    data_range = 2 ** (data_bit - 1) - 1

    data_scaled = data_float / step_size
    data_clamped = torch.clamp(data_scaled, -data_range, data_range)

    data_quantized = round_pass(data_clamped)

    if not isint:
        data_quantized = data_quantized * step_size
        quant_scale = 1.0
    if isint:
        data_quantized = data_quantized * step_size / step_size.detach()
    return data_quantized, quant_scale



def weight_quant_lsq(data_float, data_bit, step_size, isint = False):
    assert data_bit > 0
    quant_scale = (1 / step_size).detach()

    data_range = 2 ** (data_bit - 1) - 1
    gradScaleFactor = grad_scale_factor(data_range, data_float)

    step_size = grad_scale(step_size, gradScaleFactor)

    data_scaled = data_float / step_size
    data_clamped = torch.clamp(data_scaled, -data_range, data_range)

    data_quantized = round_pass(data_clamped)

    if not isint:
        data_quantized = data_quantized * step_size
        quant_scale = 1.0
    if isint:
        data_quantized = data_quantized * step_size / step_size.detach()
    return data_quantized, quant_scale


# ================================== #
# Autograd Functions
# ================================== #
# Quantize weight and add noise
def data_quant_pass_manual(data_float, data_bit, isint = False):
    assert data_bit > 1
    x_q, scale_ = DataQuant.apply(data_float, data_bit, isint)
    return x_q, scale_


class DataQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_bit, isint):
        x_q, scale = data_quant(x, data_bit,
                                isint = isint)
        scale_ = scale + 0
        # x_grad_ = abs(x_q) / (abs(x) + 1e-10)
        ctx.scale = scale
        return x_q, scale_

    @staticmethod
    def backward(ctx, feature_grad, scale_grad):
        return feature_grad * ctx.scale, None, None


class RoundPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad
# 该函数有问题，导致训练不收敛
# 直接使用torch.clamp可以避免该问题
# class DataClamp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, clamp_min, clamp_max):
#         x_clamp = torch.clamp(x, clamp_min, clamp_max)
#         ctx.grad_ = abs(x_clamp) / (abs(x) + 1e-10)
#         return x_clamp
#
#     @staticmethod
#     def backward(ctx, grad):
#         return grad * ctx.grad_, None, None
