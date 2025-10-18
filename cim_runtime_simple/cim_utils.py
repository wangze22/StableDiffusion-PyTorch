# =============================== #
# @Author  : Wang Ze
# @Time    : 2022/5/18
# @Software: PyCharm
# =============================== #
import time
import numpy as np

try:
    from c200_sdk.sdk_array_newsystem import SDKArray
except:
    pass
import math
import os
import matplotlib.pyplot as plt
import pickle as pkl
import random


# ======================================== #
# Functions related to 144K simulation
# ======================================== #
# data quant
def data_quantization(data_float, half_level = 15, isint = 0):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    if half_level <= 0:
        return data_float, 0

    data_range = abs(data_float).max()
    if data_range == 0:
        return data_float, 1

    data_quantized = (data_float / data_range * half_level).round()
    quant_scale = 1 / data_range * half_level

    if isint == 0:
        data_quantized = data_quantized * data_range / half_level
        quant_scale = 1

    return data_quantized, quant_scale


def save_pickle(file_name, dict_):
    # 获取文件夹名称
    directory = os.path.dirname(file_name)

    # 检查文件夹是否存在，如果不存在则创建
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'wb') as handle:
        pkl.dump(dict_, handle, protocol = pkl.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        dict = pkl.load(f)
    return dict


def feature_map_loss(mvm_expected, mvm_result):
    loss = (((mvm_expected - mvm_result) ** 2 / mvm_expected.size).sum()) ** 0.5 / abs(mvm_expected).max()
    return loss


# Function to plot each individual scatter plot
def scatter_plt_multi(array_output, array_output_target,
                      ax, title = 'Scatter Plot', range_abs = None):
    if range_abs is None:
        range_max = max(array_output.max(), array_output_target.max())
        range_min = min(array_output.min(), array_output_target.min())
        range_abs = max(abs(range_min), range_max)

    ax.plot(array_output_target.flatten(), array_output.flatten(),
            'o', markersize = 2, alpha = 0.1)
    ax.set_xlim(-range_abs, range_abs)
    ax.set_ylim(-range_abs, range_abs)
    ax.set_ylabel('ACIM_Output')
    ax.set_xlabel('Expected')
    ax.plot([-range_abs, range_abs], [-range_abs, range_abs], color = 'red')
    ax.axhline(0, color = 'green', linestyle = '--')
    ax.axvline(0, color = 'green', linestyle = '--')
    ax.set_aspect('equal', adjustable = 'box')
    ax.set_title(title)


def scatter_plt(array_output, array_output_target, title = 'Scatter Plot', show_fig = 1, save_fig = 0, range_abs = None):
    if range_abs is None:
        range_max = max(array_output.max(), array_output_target.max())
        range_min = min(array_output.min(), array_output_target.min())
        range_abs = max(abs(range_min), range_max)
    # 假设 array_output 和 array_output_target 是您要绘制的原始数据
    # [array_output_downsampled,
    #  array_output_target_downsampled] = downsample_data(array_output,
    #                                                     array_output_target)

    plt.figure(figsize = (6, 6))
    plt.plot(array_output_target.flatten(), array_output.flatten(),
             'o', markersize = 2, alpha = 0.1)  # Use plt.plot instead of plt.scatter
    plt.xlim(-range_abs, range_abs)
    plt.ylim(-range_abs, range_abs)
    plt.ylabel('array_output')
    plt.xlabel('array_output_target')
    plt.plot([-range_abs, range_abs], [-range_abs, range_abs], color = 'red')
    plt.axhline(0, color = 'green', linestyle = '--')
    plt.axvline(0, color = 'green', linestyle = '--')
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.title(title)
    if save_fig:
        save_path = 'plt'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, f'{title}.png')
        plt.savefig(file_path)
        print(f'{file_path} saved!')
    if show_fig:
        plt.show()
    plt.close()


# 计算网络推理的 softmax, 即概率
def softmax(input_data):
    input_data = input_data.squeeze()
    max_num = input_data.max()
    input_data -= max_num
    prob = np.exp(input_data.max()) / np.exp(input_data).sum()
    return prob


def input_multi_bits_shift_expansion(input_matrix, dac_bits = 2):
    input_matrix = np.round(input_matrix).astype(np.int64)

    if (input_matrix == 0).all():
        return input_matrix, 1

    rows, cols = input_matrix.shape
    input_matrix = input_matrix.T.flatten()

    shift_value = (1 << (dac_bits - 1)) - 1
    input_bits = math.floor(math.log2(np.max(np.abs(input_matrix)))) + 1
    max_expansion_times = math.ceil(input_bits / (dac_bits - 1))

    input_expanded = np.zeros((rows * cols, max_expansion_times), dtype = np.int8)
    input_matrix_sign = np.sign(input_matrix)
    input_matrix = np.abs(input_matrix)

    for i in range(max_expansion_times):
        pulse_cur = (input_matrix & shift_value) * input_matrix_sign
        input_expanded[:, i] = pulse_cur
        input_matrix >>= (dac_bits - 1)

    input_expanded = input_expanded.reshape(cols, rows, max_expansion_times).transpose(1, 0, 2).reshape(rows, -1)
    return input_expanded, max_expansion_times


def restore_shift_expansion_output(output, bitlen, dac_bits, output_bitwise_batch):
    [cal_times, output_cols] = output.shape
    # 对计算结果按照展开的位数进行求和
    output_bitwise_row = 0
    output_row = 0
    if bitlen == 0:
        output[output_row:output_row + cal_times, :] = 0
    else:
        factor_list = np.array([2 ** (i * (dac_bits - 1)) for i in range(bitlen)] * cal_times)
        factor_list = factor_list.reshape(bitlen * cal_times, -1)
        output_temp = output_bitwise_batch[output_bitwise_row:
                                           output_bitwise_row + cal_times * bitlen, :
                      ] * factor_list
        output_temp = output_temp.reshape(cal_times, bitlen, output_cols)
        output[output_row:output_row + cal_times, :] = output_temp.sum(axis = 1)
    output_row += cal_times
    return output


# MVM矩阵乘法计算函数
def mvm_calculate(chip_idx, input_matrix, addr, weight, it_time = 5,
                  use_simulator = False, dac_bits = 2, adc_bits = 4,
                  adc_scale = 1 / 63, noise_scale = 0.05):

    # chip_idx：芯片编号
    # input_matrix：输入矩阵
    # addr：权重地址
    # it_time：ADC积分时间
    # use_simulator：是否使用仿真器。当手头没有忆阻器芯片时，打开此选项。仿真器的计算流程与芯片一致。
    input_matrix[input_matrix > 127] = 127
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input_matrix.shape[1]
    # 输出通道数
    output_cols = addr[3]
    output = np.zeros([cal_times, output_cols])

    # 输入数据进行2进制编码，并逐次展开
    # t = time.time()
    input_expanded, max_expansion_times = input_multi_bits_shift_expansion(input_matrix,
                                                                           dac_bits = dac_bits)
    # t = time.time() - t
    # print(f'Software Expansion Time = {t:.3f}s ({dac_bits}-bit DAC)')

    # ADC增益为1时的缩放系数
    ADC_scale = adc_scale * it_time

    # 使用仿真器计算
    t = time.time()
    if use_simulator:
        # ADC偏置噪声系数
        offset_noise = np.random.randn(output_cols) * 15 * noise_scale * it_time / 63

        if weight is None:
            chip_data = np.load(f'chip_idx_{chip_idx}.npy')
            weight = chip_data[addr[0]:addr[0] + addr[2], addr[1]:addr[1] + addr[3]]
        w_range = weight.max() - weight.min()
        shape = weight.shape
        w_noise = w_range * noise_scale * np.random.randn(*shape)
        # print(f'noise_mean = {w_noise.mean():.5f}')
        weight_n = weight + w_noise

        # MVM计算，加入了噪声模拟
        ADC_output = np.dot(input_expanded.transpose(1, 0), weight_n) + offset_noise
        # 仿真模拟计算的缩放效应
        ADC_output *= ADC_scale
        # 模拟ADC的量化
        ADC_output = np.round(ADC_output)
        # ADC动态范围截断
        ADC_range = 2 ** (adc_bits - 1) - 1
        ADC_output[ADC_output > ADC_range] = ADC_range
        ADC_output[ADC_output < -ADC_range] = -ADC_range

    # 使用忆阻器芯片计算
    else:
        sdk = SDKArray(chip_idx)
        ADC_output = sdk.calculate(input_expanded.transpose(1, 0),
                                   addr, it_time = it_time)
    t = time.time() - t

    # 输出数据移位累加
    output = restore_shift_expansion_output(output = output, bitlen = max_expansion_times,
                                            dac_bits = dac_bits, output_bitwise_batch = ADC_output)

    return output, ADC_output, ADC_scale


def generate_weight_cfg_image(weight_cfg, array_size):
    def generate_bright_color():
        """
        Generates a bright, saturated color by ensuring one component is fully saturated (1.0),
        while the other components are at least 0.5.
        """
        color = [0, 0, 0]
        primary_index = random.randint(0, 2)
        color[primary_index] = 1.0
        for i in range(3):
            if i != primary_index:
                color[i] = random.uniform(0.5, 1.0)
        return color

    # Determine scaling factor based on minimum size requirement (1000 pixels on the shortest side)
    min_target_size = 1000
    rows, cols = array_size
    min_size = min(rows, cols)
    scaling_factor = max(min_target_size / min_size, 1)
    scaled_rows = int(rows * scaling_factor)
    scaled_cols = int(cols * scaling_factor)

    # Create a black background image
    fig, ax = plt.subplots(figsize = (scaled_cols / 100, scaled_rows / 100), dpi = 100)

    # Set up the plot appearance
    ax.set_xlim([0, scaled_cols])
    ax.set_ylim([scaled_rows, 0])  # Inverted Y-axis to match image coordinates
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis("on")
    ax.tick_params(axis = 'both', colors = 'white', labelsize = 12)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.grid(color = 'gray', linestyle = '-', linewidth = 0.5)

    # Set ticks and labels
    xticks = np.arange(0, scaled_cols + 1, scaled_cols // 10)
    yticks = np.arange(0, scaled_rows + 1, scaled_rows // 10)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{int(x / scaling_factor)}" for x in xticks], color = 'white')
    ax.set_yticklabels([f"{int(y / scaling_factor)}" for y in yticks], color = 'white')

    # Draw an outer border around the entire array
    border_color = 'white'
    border_width = 5
    outer_rect = plt.Rectangle((0, 0), scaled_cols, scaled_rows,
                               linewidth = border_width,
                               edgecolor = border_color, facecolor = 'none')
    ax.add_patch(outer_rect)

    # Draw boxes and labels for each layer in weight_cfg
    for layer_name, addr in weight_cfg.items():
        x0, y0, height, width = [int(coord * scaling_factor) for coord in addr]
        x1 = x0 + height
        y1 = y0 + width
        color = generate_bright_color()

        # Draw the rectangle
        rect = plt.Rectangle((y0, x0), width, height, linewidth = 5, edgecolor = color, facecolor = 'none')
        ax.add_patch(rect)

        # Add the label with larger font size
        label_x = y0 + width // 2
        label_y = x0 + height // 2
        ax.text(label_x, label_y, layer_name, color = color, ha = 'center', va = 'center', fontsize = 20, fontweight = 'bold')

    plt.savefig("weight_cfg_image.png", bbox_inches = 'tight', pad_inches = 0, dpi = 100)
    plt.close()
