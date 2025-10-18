# from sim_utils import *
import colorsys
import copy
import importlib.util
import json
import math
# import pandas as pd
import os
import pickle as pkl
import types

import matplotlib.pyplot as plt
import torch.nn as nn

import numpy as np
import torch
import onnx
from onnx import shape_inference
import pandas as pd

def to_numpy(data):
    # 检查输入数据是否为 PyTorch 张量
    if isinstance(data, torch.Tensor):
        # 检查张量是否需要 detach
        if data.requires_grad:
            data = copy.deepcopy(data.detach())
        # 确保张量在 CPU 上
        if data.is_cuda:
            data = copy.deepcopy(data.cpu())
        # 转换为 NumPy 数组
        numpy_array = data.numpy()

    # 检查输入数据是否为 NumPy 数组
    elif isinstance(data, np.ndarray):
        # 直接返回 NumPy 数组
        numpy_array = data

    else:
        raise TypeError("输入数据既不是 PyTorch 张量也不是 NumPy 数组")

    return numpy_array

def to_tensor(data, device):
    # 检查数据是否为 PyTorch 张量
    if isinstance(data, torch.Tensor):
        return data.to(device)
    # 检查数据是否为 NumPy 数组
    elif isinstance(data, np.ndarray):
        # 转换 NumPy 数组为 PyTorch 张量
        return torch.from_numpy(data.copy()).to(device)
    else:
        raise TypeError("输入数据类型必须是 NumPy 数组或者 PyTorch 张量")

def has_abnormal_values(array):
    # 判断是否存在inf
    has_inf = torch.isinf(array).any().item()

    # 判断是否存在nan
    has_nan = torch.isnan(array).any().item()

    if has_inf or has_nan:
        return True
    else:
        return False

def read_json_to_dict(filepath):
    # 打开文件并读取数据
    with open(filepath, 'r') as file:
        data_dict = json.load(file)
    return data_dict


def save_tensors_to_csv(file_name, *tensors):
    """
    将多个 PyTorch Tensor 拉成列向量，并保存为 CSV 文件。

    参数:
    - file_name (str): 输出的 CSV 文件名。
    - *tensors: 多个 PyTorch Tensor，每个 Tensor 将保存为一列。

    返回:
    - None
    """
    data = {}
    for idx, tensor in enumerate(tensors):
        # 检查输入是否是 PyTorch Tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"输入的第 {idx + 1} 个参数不是一个 PyTorch Tensor。")

        # 将 Tensor 拉成一维列向量
        tensor_column = tensor.reshape(-1).cpu().numpy()
        data[f'Column_{idx + 1}'] = tensor_column

    # 将数据保存为 DataFrame 并导出为 CSV
    df = pd.DataFrame(data)
    # 获取文件保存目录
    directory = os.path.dirname(file_name)
    # 检查文件夹是否存在，如果不存在则创建
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    df.to_csv(file_name, index = False)
    print(f"已成功保存至 {file_name}")


def save_to_json(dictionary, filename):
    def format_value(value):
        if type(value).__name__ == 'RepeatedScalarContainer':
            return list(value)
        if isinstance(value, dict):
            return {k: format_value(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            dims = value.shape
            return f"tensor, dim = {dims}"
        elif hasattr(value, '__class__') and value.__class__.__name__ == 'cim_tensor':
            # 直接返回数据的类型，而不是value
            return f"cim_tensor, dim = {value.shape}"
        elif isinstance(value, torch.nn.Module):
            module_type = value.__class__.__name__
            return f"PyTorch Module, type = {module_type}"
        else:
            return value

    def handle_non_serializable(obj):
        # 对于不可序列化的对象，返回它们的类型
        return f"{obj.__class__.__name__}, not serializable"

    formatted_dict = format_value(dictionary)
    #
    # 获取文件保存目录
    directory = os.path.dirname(filename)
    # 检查文件夹是否存在，如果不存在则创建
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w') as file:
        json.dump(formatted_dict, file, indent = 4, default = handle_non_serializable)

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

def load_json_as_dict(file_path):
    """
    读取一个JSON文件，并将其内容转换为字典返回。

    参数:
    file_path (str): JSON文件的路径。

    返回:
    dict: 读取的JSON内容转换后的字典。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 读取每层计算的py配置文件,以字典形式传回
def load_config_from_py(py_path):
    spec = importlib.util.spec_from_file_location("config_module", py_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # 获取基本类型的变量，过滤掉模块、类、函数等
    config_dict = {}
    for k, v in config_module.__dict__.items():
        if not k.startswith('__') \
                and not callable(v) \
                and not isinstance(v, type) \
                and not isinstance(v, types.ModuleType):
            config_dict[k] = v

    return config_dict


def change_dict_value(d, key_change, value_change):
    for key in d.keys():
        if key == key_change:
            d[key] = value_change
        elif isinstance(d[key], dict):
            change_dict_value(d[key], key_change, value_change)
        elif isinstance(d[key], list):
            for item in d[key]:
                if isinstance(item, dict):
                    change_dict_value(item, key_change, value_change)
    return d


def get_dict_value(d, key_target, value_list = None):
    if value_list is None:
        value_list = []
    for key, value in d.items():
        if key == key_target:
            value_list.append(value)
        elif isinstance(value, dict):
            get_dict_value(value, key_target, value_list)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    get_dict_value(item, key_target, value_list)
    return value_list


def flatten_list(nested_list):
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list


def scatter_plt(array_output_, array_output_target_,
                title = 'Scatter Plot', show_fig = 1, save_fig = 0, save_path = 'scatter_plot',
                range_abs = None, col_color = False):
    array_output = to_numpy(array_output_)
    array_output_target = to_numpy(array_output_target_)
    alpha = min(1.0, 1 / math.log10(array_output.size))
    # alpha = max(0.1, alpha)

    def generate_colors(num_colors, saturation = 0.9, lightness = 0.6):
        colors = []
        for i in np.linspace(0, 1, num_colors, endpoint = False):
            hue = i
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)
        return colors

    if range_abs is None:
        range_max = max(array_output.max(), array_output_target.max())
        range_min = min(array_output.min(), array_output_target.min())
        range_abs = max(abs(range_min), range_max)

    plt.figure(figsize = (6, 6))

    if col_color:
        num_cols = array_output.shape[1]
        colors = generate_colors(num_cols)
        for col in range(num_cols):
            try:
                plt.plot(array_output_target[:, col].flatten(), array_output[:, col].flatten(),
                         'o', markersize = 2, alpha = min(1.0, alpha * num_cols), color = colors[col])
            except:
                x = 1
    else:
        plt.plot(array_output_target.flatten(), array_output.flatten(),
                 'o', markersize = 2, alpha = alpha)

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
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, f'{title}.png')
        plt.savefig(file_path)
        print(f'{file_path} saved!')

    if show_fig:
        plt.show()

    plt.close()

def export_onnx(model,input_data, onnx_path = 'export_model.onnx', device = 'cuda'):
    # 获取目录路径
    dir_path = os.path.dirname(onnx_path)
    # 检查目录是否存在，如果不存在，则创建它
    if not os.path.exists(dir_path) and len(dir_path) > 0:
        os.makedirs(dir_path)

    model.eval()

    def to_device(data):
        if isinstance(data, list):
            return [item.to(device) if isinstance(item, torch.Tensor) else item for item in data]
        elif isinstance(data, tuple):
            return tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    input_data = to_device(input_data)
    model.to(device)
    model_path = f"{onnx_path}.onnx"
    torch.onnx.export(model,  # model being run
                      input_data,  # model input (or a tuple for multiple inputs)
                      model_path,  # where to save the model
                      export_params = True,  # store the trained parameter weights inside the model file
                      opset_version = 11,  # the ONNX version to export the model to
                      do_constant_folding = True,  # whether to execute constant folding for optimization
                      input_names = ['modelInput'],  # the model's input names
                      output_names = ['modelOutput'],  # the model's output names
                      # keep_initializers_as_inputs = True
                      # export_modules_as_functions = True
                      )
    inferred_model = onnx.load_model(model_path)
    inferred_model = shape_inference.infer_shapes(inferred_model)
    onnx.save(inferred_model, model_path)


def forward_with_hooks_layer_flag(model, input_data):
    output_dict = {}
    hooks = []

    def hook_factory(name):
        def hook(module, input, output):
            output_dict[name] = output

        return hook

    for name, module in model.named_modules():
        hook = hook_factory(name)
        hooks.append(module.register_forward_hook(hook))

    output = model(input_data)

    for hook in hooks:
        hook.remove()

    return output, output_dict

import textwrap
import matplotlib.pyplot as plt
import textwrap

import matplotlib.pyplot as plt
import textwrap


def plot_loss(loss_list, title = 'Loss', file_path = 'loss_plot.png'):
    # 创建一个临时图形以获取宽度
    fig, ax = plt.subplots()
    ax.plot(loss_list)

    # 获取当前图形的宽度（以像素为单位）
    fig_width_in_inches = fig.get_size_inches()[0]
    dpi = fig.dpi
    fig_width_in_pixels = fig_width_in_inches * dpi

    # 计算标题的最大宽度（80% 的图像宽度）
    max_title_width = fig_width_in_pixels * 0.8

    # 估算字符的平均宽度（一个近似值）
    average_char_width = 7  # 可以根据具体字体大小调整

    # 计算合适的换行宽度（字符数）
    max_title_chars = int(max_title_width / average_char_width)

    # 使用 textwrap 自动换行
    wrapped_title = "\n".join(textwrap.wrap(title, width = max_title_chars))

    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 绘制图形并保存
    plt.plot(loss_list)
    plt.title(wrapped_title)
    plt.savefig(file_path)
    plt.close(fig)
    print(f'{file_path} saved!')
