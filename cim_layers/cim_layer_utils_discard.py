import torch
import torch.nn as nn
import numpy as np

def weight_reshape(weight_data,
                   weight_arrange_mode,
                   weight_type):
    if weight_type == 'Conv2d':
        # C_out, C_in, H_kernel, W_kernel = weight_data.shape
        weight_reshape_2D = conv_kernel_auto_reshape(weight_data, weight_arrange_mode)
        return weight_reshape_2D
    # 全连层权重
    elif weight_type == 'Linear':
        weight_reshape_2D = weight_data.permute(1, 0)
        return weight_reshape_2D


def weight_transpose_order(weight_arrange_mode):
    # 定义一个映射，将字符映射到对应的轴
    arrange_map = {'C': 1, 'H': 2, 'W': 3}

    # 解析weight_arrange_mode，转换成transpose需要的轴顺序
    transpose_order = [0]  # 第一个轴(C_out)保持不变
    for char in weight_arrange_mode:
        if char in arrange_map:
            transpose_order.append(arrange_map[char])
    return transpose_order


def conv_kernel_auto_reshape(weight_reshape_2D, weight_arrange_mode):
    C_out, C_in, H_kernel, W_kernel = weight_reshape_2D.shape
    transpose_order = weight_transpose_order(weight_arrange_mode)
    weight_reshape_2D = weight_reshape_2D.permute(transpose_order)
    weight_reshape_2D = weight_reshape_2D.reshape(C_out, -1).transpose(0, 1)
    return weight_reshape_2D

def feature_map_to_input_auto_transpose(feature_map, kernel_size, stride, padding,
                                        weight_arrange_mode):
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    batch, in_channels, feature_in_h, feature_in_w = feature_map.shape
    feature_out_w = int((feature_in_w - kernel_size + 2 * padding) / stride + 1)
    feature_out_h = int((feature_in_h - kernel_size + 2 * padding) / stride + 1)
    feature_map = feature_map_padding(feature_map, padding)
    input_rows = kernel_size ** 2 * in_channels
    slides = feature_out_w * feature_out_h
    array_input = torch.zeros([batch, input_rows, slides], device = feature_map.device)
    slide_idx = 0
    for i in range(feature_out_h):
        for j in range(feature_out_w):
            slide_window = feature_map[:, :, i * stride:i * stride + kernel_size,
                           j * stride:j * stride + kernel_size]
            transpose_order = feature_map_transpose_order(weight_arrange_mode)
            array_input[:, :, slide_idx] = slide_window.transpose(transpose_order).reshape(batch, -1)
            slide_idx += 1
    return array_input

def feature_map_padding(feature_map, padding):
    # feature_map 维度： B, C, H, W
    if padding > 0:
        feature_map = np.pad(feature_map,
                             pad_width = ((0, 0), (0, 0),
                                          (padding, padding),
                                          (padding, padding)),
                             mode = 'constant', constant_values = 0)
    return feature_map

def feature_map_transpose_order(weight_arrange_mode):
    # 定义一个映射，将字符映射到对应的轴
    arrange_map = {'C': 1, 'H': 2, 'W': 3}

    # 解析weight_arrange_mode，转换成transpose需要的轴顺序
    transpose_order = [0]
    for char in weight_arrange_mode:
        if char in arrange_map:
            transpose_order.append(arrange_map[char])
    return transpose_order


def feature_map_to_input(feature_map, kernel_size, stride, padding,
                         weight_arrange_mode):
    if weight_arrange_mode == 'CHW':
        return feature_map_to_input_torch(feature_map, kernel_size, stride, padding)
    else:
        return feature_map_to_input_auto_transpose(feature_map, kernel_size, stride, padding,
                                                   weight_arrange_mode)


def feature_map_to_input_torch(feature_map, kernel_size, stride, padding):
    unfold = nn.Unfold(kernel_size, padding = padding, stride = stride)
    while (len(feature_map.shape) < 4):
        feature_map = feature_map.unsqueeze(dim = 0)
    array_input = unfold(feature_map.float())
    return array_input

def output_to_feature_map(out_put, out_w, out_h):
    # out_put shape = [W_out * H_out, C_out]
    # feature_map shape = [C_out, W_out, H_out]
    batch_num = out_put.shape[0]
    channels = out_put.shape[2]
    feature_map = out_put.permute(0, 2, 1).reshape([batch_num, channels, out_w, out_h])
    # feature_map = np.expand_dims(feature_map, 0)
    return feature_map