# =============================== #
# @Author  : Wang Ze
# @Time    : 2022/5/18
# @Software: PyCharm
# =============================== #
from .cim_utils import *
from .cim_adc import *
import matplotlib.pyplot as plt


# ======================= #
# 函数形式计算层
# ======================= #
# 给 feature_map 加上 padding
def feature_map_padding(feature_map, padding):
    # feature_map 维度： C, W, H
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    feature_map_pad = np.pad(feature_map,
                             ((0, 0),
                              (padding, padding),
                              (padding, padding)),
                             mode = 'constant')
    return feature_map_pad


# 将忆阻器每层的输出 out_put 转换回 feature_map 的形式
def output_to_feature_map(out_put, out_w, out_h):
    # out_put shape = [W_out * H_out, C_out]
    # feature_map shape = [C_out, W_out, H_out]
    channels = out_put.shape[1]
    feature_map = out_put.transpose(1, 0).reshape([channels, out_w, out_h])
    return feature_map


# 将 feature_map 转化为下一层忆阻器的输入 array_input
def feature_map_to_input(feature_map, kernel_size, stride, padding, repeat = None):
    # feature_map shape = [C_in, W_in, H_in]
    # array_input shape = [W_out * H_out, C_out]
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    in_channels = feature_map.shape[0]
    feature_in_w = feature_map.shape[1]
    feature_in_h = feature_map.shape[2]
    feature_out_w = int((feature_in_w - kernel_size + 2 * padding) / stride + 1)
    feature_out_h = int((feature_in_h - kernel_size + 2 * padding) / stride + 1)
    feature_map = feature_map_padding(feature_map, padding)
    input_rows = kernel_size ** 2 * in_channels
    output_rows = feature_out_w * feature_out_h
    array_input = np.zeros([input_rows, output_rows])
    idx = 0
    for i in range(feature_out_w):
        for j in range(feature_out_h):
            slide_window = feature_map[:, i * stride:i * stride + kernel_size,
                           j * stride:j * stride + kernel_size]
            array_input[:, idx] = slide_window.reshape(-1)
            idx += 1
    if repeat:
        array_input = np.tile(array_input, [repeat[0], 1])
    return array_input


# 池化 fast
def pooling(feature_map, kernel_size):
    channels = feature_map.shape[0]
    pooled_rows = int(feature_map.shape[1] / kernel_size)
    pooled_cols = int(feature_map.shape[2] / kernel_size)
    output = feature_map.reshape(channels, pooled_rows,
                                 kernel_size, pooled_cols, kernel_size)
    output = output.max(axis = (2, 4))
    return output


# 144k 片上推理卷积封装 函数形式
def conv2d_144k(chip_idx, input_feature_map, weight_addr, repeat,
                stride, kernel_size, padding,
                input_half_level, output_half_level,
                it_time = 10,
                adjust_ADC = False,
                relu = True,
                input_quant = False):
    # ================================= #
    # 参数说明
    # ================================= #
    # input_feature_map:
    #   输入 feature map, 矩阵大小为 [C, H, W]
    # weight_addr:
    #   144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    # repeat:
    #   144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    # stride, kernel_size, padding：
    #   卷积相关参数
    # input_half_level:
    #   输入数据量化等级
    # output_half_level:
    #   输出数据量化等级
    # relu:
    #   是否 relu
    # input_quant:
    #   是否对输入数据进行量化

    # 补齐维度
    while len(input_feature_map.shape) < 3:
        input_feature_map = np.expand_dims(input_feature_map, axis = 0)

    # 计算输出大小
    _, input_rows, input_cols = input_feature_map.shape
    out_feature_size_rows = int((input_rows + 2 * padding - kernel_size) / stride + 1)
    out_feature_size_cols = int((input_cols + 2 * padding - kernel_size) / stride + 1)

    # 输入数据量化
    if input_quant:
        input_feature_map, _ = data_quantization(input_feature_map,
                                                 half_level = input_half_level, isint = 1)
    # 输入图像重排
    array_input = feature_map_to_input(input_feature_map, stride = stride,
                                       kernel_size = kernel_size,
                                       padding = padding, repeat = repeat)
    if adjust_ADC:
        it_time = ADC_auto_adjust(chip_idx = chip_idx,
                                  input_matrix = array_input,
                                  addr = weight_addr,
                                  target_percent = [0.01, 0.02], verbose = 0)
    # 乘加运算(卷积 f1), 使用脉冲展开
    mvm_result, _, _ = mvm_calculate(chip_idx,
                                     input_matrix = array_input,
                                     addr = weight_addr,
                                     it_time = it_time)

    # Relu
    if relu:
        mvm_result[mvm_result < 0] = 0
    # 数据量化
    mvm_result, _ = data_quantization(mvm_result,
                                      half_level = output_half_level, isint = 1)
    # 数据重排
    layer_output = output_to_feature_map(mvm_result,
                                         out_feature_size_rows,
                                         out_feature_size_cols)
    if adjust_ADC:
        return layer_output, it_time
    return layer_output


# 144k 片上推理全连接封装 函数形式
def linear_144k(chip_idx, input_feature_map, weight_addr, repeat,
                input_half_level, output_half_level,
                it_time = 10,
                adjust_ADC = False,
                relu = True,
                input_quant = False):
    # ================================= #
    # 参数说明
    # ================================= #
    # input_feature_map:
    #   输入 feature map, 矩阵大小为 [C, H, W]
    # weight_addr:
    #   144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    # repeat:
    #   144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    # input_half_level:
    #   输入数据量化等级
    # output_half_level:
    #   输出数据量化等级
    # relu:
    #   是否 relu
    # input_quant:
    #   是否对输入数据进行量化
    array_input = input_feature_map.reshape(-1, 1)
    if input_quant:
        array_input, _ = data_quantization(array_input, half_level = input_half_level, isint = 1)
    array_input = np.tile(array_input, [repeat[0], 1])

    if adjust_ADC:
        it_time = ADC_auto_adjust(chip_idx = chip_idx,
                                  input_matrix = array_input,
                                  addr = weight_addr,
                                  target_percent = [0.01, 0.02],
                                  verbose = 0)
    mvm_result, _, _ = mvm_calculate(chip_idx,
                                     input_matrix = array_input,
                                     addr = weight_addr,
                                     it_time = it_time)
    if relu:
        mvm_result[mvm_result < 0] = 0
    layer_output, _ = data_quantization(mvm_result, half_level = output_half_level, isint = 1)
    if adjust_ADC:
        return layer_output, it_time
    return layer_output
