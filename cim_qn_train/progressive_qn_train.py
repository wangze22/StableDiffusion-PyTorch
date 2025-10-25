# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

import onnx
import torch
import torch.nn.functional as F
import torch.optim as optim
from onnx import shape_inference
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
import cim_layers.register_dict as reg_dict
import cim_qn_train.layers_enhance as en
# import cim_runtime.cim_top as ct
from cim_layers.layers_all import *
from cim_toolchain_utils.utils import *
from . import hybrid_train_tools as hbt
import sys
import inspect
from cim_qn_train.train_utils import mvm_time_est_144k


class ProgressiveTrain():
    def __init__(self, model, device = None, device_ids = None, name = 'model'):
        self.model = model
        self.model_name = name
        # ======================== #
        # Quant
        # ======================== #
        self.weight_bit = 8

        self.input_bit = 8
        self.output_bit = 8

        self.input_quant = True
        self.output_quant = True
        self.weight_quant = True
        # ======================== #
        # Clamp
        # ======================== #
        self.clamp_std = 0
        # ======================== #
        # Noise
        # ======================== #
        self.noise_scale = 0
        self.gain_noise_scale = 0
        self.offset_noise_scale = 0
        # ======================== #
        # ADDA
        # ======================== #
        self.adc_bit = 8
        self.dac_bit = 8
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        self.device_ids = device_ids
        self.onnx_model = None
        # ======================== #
        # Parameter Registration
        # ======================== #
        self.para_list = ['weight_bit', 'input_bit', 'output_bit',
                          'input_quant', 'output_quant', 'weight_quant',
                          'clamp_std',
                          'noise_scale', 'gain_noise_scale', 'offset_noise_scale',
                          'adc_bit', 'dac_bit']
        self.assign_module_name()

    def assign_module_name(self):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.op_layers:
                module.name = name

    def train_model(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def model_info(self):
        return self.get_model_info()

    def get_model_info(self, weight_info = False, show_nn_layers = True):
        model_info_dict = {}
        for name, module in self.model.named_modules():
            # 跳过整个模型本身的统计，只统计子模块
            if name == "":
                continue
            # 判断是否为最末层模块，通过检查它是否有子模块
            # 注意：我们使用list转换generator为列表，然后检查长度
            if type(module) in reg_dict.op_layers:
                if show_nn_layers and not type(module) in reg_dict.op_layers:
                    continue
                model_info_dict[name] = {}
                model_info_dict[name]['type'] = type(module).__name__
                model_info_dict[name]['layer_flag'] = getattr(module, 'layer_flag', 'regular_layer')
                weight = getattr(module, 'weight', None)
                if weight is not None:
                    weight_shape = weight.shape
                    model_info_dict[name]['weight_shape'] = list(weight_shape)
                if weight_info:
                    model_info_dict[name]['weights'] = weight

        return model_info_dict

    def get_parameters_count(self):
        param_count = {}
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.op_layers:
                params = sum(p.numel() for p in module.parameters())
                # 如果 module 类型在 reg_dict.op_layers 中
                if type(module) in reg_dict.op_layers:
                    param_count[type(module).__name__] = param_count.get(type(module).__name__, 0) + params
                # 总参数数目
                param_count['total_parameters'] = param_count.get('total_parameters', 0) + params
        return param_count

    @property
    def layer_names(self):
        name_list = []
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.op_layers:
                name_list.append(name)
        return name_list

    #
    # @property
    # def qn_layers(self):
    #     name_list = []
    #     for name, module in self.model.named_modules():
    #         if type(module) in (l_qn_lsq.Conv2d_qn_lsq, l_qn_lsq.Linear_qn_lsq):
    #             name_list.append(name)
    #     return name_list
    #
    # @property
    # def adda_layers(self):
    #     name_list = []
    #     for name, module in self.model.named_modules():
    #         if type(module) in (adda.Conv2d_ADDA_aware, adda.Linear_ADDA_aware):
    #             name_list.append(name)
    #     return name_list
    #
    # @property
    # def torch_layers(self):
    #     name_list = []
    #     for name, module in self.model.named_modules():
    #         if type(module) in (nn.Conv2d, nn.Linear):
    #             name_list.append(name)
    #     return name_list
    #
    # @property
    # def cim_layers(self):
    #     name_list = []
    #     for name, module in self.model.named_modules():
    #         if type(module) in (cl.Conv2d_cim, cl.Linear_cim):
    #             name_list.append(name)
    #     return name_list

    @property
    def customized_layers(self):
        name_dict = {}
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_layers:
                type_name = type(module).__name__
                if type_name not in name_dict:
                    name_dict[type_name] = []
                name_dict[type_name].append(name)
        return name_dict

    def cal_ops(self, module, input_size):
        """
        计算给定模块的乘加操作数
        参数:
        module (nn.Module): 要计算的模块
        input_data (Tensor): 模块的输入数据

        返回:
        ops (int): 模块的乘加操作数
        """
        ret_dict = {}
        if type(module) in reg_dict.conv_layers:
            # 卷积层的计算量
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups

            # 输入数据的形状
            batch_size, _, input_height, input_width = input_size

            # 输出数据的形状
            output_height = (input_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            output_width = (input_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1

            # 每个卷积核的计算量（乘法和加法分别计算）
            ops_per_filter_mult = (in_channels // groups) * kernel_size[0] * kernel_size[1]
            ops_per_filter_add = ops_per_filter_mult - 1

            # 所有卷积核的计算量
            add_ops = batch_size * output_height * output_width * out_channels * ops_per_filter_add
            mul_ops = batch_size * output_height * output_width * out_channels * ops_per_filter_mult
            total_ops = add_ops + mul_ops
            mvm_times = batch_size * output_height * output_width

        elif type(module) in reg_dict.linear_layers:
            # 全连接层的计算量
            in_features = module.in_features
            out_features = module.out_features

            # 输入数据的形状
            batch_size = input_size[0]

            # 每个神经元的计算量（乘法和加法分别计算）
            ops_per_neuron_mult = in_features
            ops_per_neuron_add = in_features - 1

            # 所有神经元的计算量
            add_ops = batch_size * out_features * ops_per_neuron_add
            mul_ops = batch_size * out_features * ops_per_neuron_mult
            total_ops = add_ops + mul_ops
            mvm_times = batch_size

        else:
            add_ops = 0
            mul_ops = 0
            total_ops = 0
            mvm_times = 0

        ret_dict['add_ops'] = add_ops
        ret_dict['mul_ops'] = mul_ops
        ret_dict['total_ops'] = total_ops
        ret_dict['mvm_times'] = mvm_times
        return ret_dict

    def get_energy_dict(self,
                        input_data,
                        tops_j_dmac = 5,
                        tops_j_acim = 50,
                        size_acim = [256, 256],
                        size_dmac = [64, 1],
                        ):
        acim_area = 1
        for num in size_acim:
            acim_area *= num
        dmac_area = 1
        for num in size_dmac:
            dmac_area *= num
        tops_acim_per_cal = (acim_area * 2 - size_acim[1]) / 1e12
        tops_dmac_per_cal = (dmac_area * 2 - size_dmac[1]) / 1e12
        # tops_acim_per_cal = (acim_area * 2 ) / 1e12
        # tops_dmac_per_cal = (dmac_area * 2 ) / 1e12

        ops_dict = self.gen_ops_dict(input_data)
        energy_acim_per_cal = tops_acim_per_cal / tops_j_acim
        energy_dmac_per_cal = tops_dmac_per_cal / tops_j_dmac
        # ========================= #
        # ACIM 能耗 / 计算量
        # ========================= #
        tot_energy_acim = 0
        tot_ops_acim = 0
        # ========================= #
        # DMAC 能耗 / 计算量
        # ========================= #
        # 包含所有需要数字计算的层，例如 adapter，或者主干网络中没有使用 ACIM 计算的层
        tot_energy_dmac = 0
        tot_ops_dmac = 0
        # ========================= #
        # Adapter 能耗 / 计算量
        # ========================= #
        # Adapter 计算量是 DMAC 的子集，所有 adapter 都需要数字计算核
        tot_energy_adapter = 0
        tot_ops_adapter = 0
        # ========================= #
        # 每层数据
        # ========================= #
        layer_dict = {}
        for name, module in self.model.named_modules():
            # Check if the current module is a convolutional or linear layer, and replace accordingly
            if name in ops_dict:
                mvm_times = ops_dict[name]['mvm_times']
                layer_ops = ops_dict[name]['total_ops']
                if type(module) in reg_dict.cim_layers:
                    # ------------------------------------------------- #
                    # 根据 mapping 信息统计权重在 ACIM 上拆分的次数
                    acim_cal_times_mapping = len(module.weight_mapping_info) * mvm_times
                    # ------------------------------------------------- #
                    # 简单手动计算权重在 ACIM 上拆分的次数
                    weight = module.weight
                    weight_2d = weight.data.reshape(weight.shape[0], -1).transpose(0, 1)
                    rows, cols = weight_2d.shape
                    acim_cal_times = math.ceil(rows / size_acim[0]) * math.ceil(cols / size_acim[1]) * mvm_times
                    # ------------------------------------------------- #
                    assert acim_cal_times_mapping == acim_cal_times
                    # ------------------------------------------------- #
                    layer_energy = energy_acim_per_cal * acim_cal_times
                    tot_energy_acim += layer_energy
                    # ============================== #
                    # 以下有两种计算 ops 的方法
                    # 第一种不统计权重拆分成多个阵列后 mapping 之间的加法
                    # 第二种统计，但是在进行 tops/w 的计算时，有可能超出峰值算力
                    # ============================== #
                    # 方法1 （最终结果不会超过峰值算力）：
                    layer_ops = layer_ops - (math.ceil(rows / size_acim[0]) - 1) * cols * mvm_times
                    tot_ops_acim += layer_ops
                    # 方法2：
                    # tot_ops_acim += layer_ops
                    # ============================== #
                    layer_topsw = layer_ops / (energy_acim_per_cal * acim_cal_times) / 1e12
                    assert layer_topsw <= tops_j_acim
                    # ============================== #
                    layer_dict[name] = {}
                    layer_dict[name]['device'] = 'ACIM'
                    layer_dict[name]['energy'] = layer_energy
                    layer_dict[name]['splits'] = len(module.weight_mapping_info)
                    layer_dict[name]['ops'] = layer_ops
                    layer_dict[name]['tops/w'] = layer_topsw
                    continue
                if type(module) in reg_dict.op_layers:
                    weight = module.weight
                    weight_2d = weight.data.reshape(weight.shape[0], -1).transpose(0, 1)
                    rows, cols = weight_2d.shape
                    dmac_cal_times = math.ceil(rows / size_dmac[0]) * math.ceil(cols / size_dmac[1]) * mvm_times
                    layer_energy = energy_dmac_per_cal * dmac_cal_times
                    tot_energy_dmac += layer_energy
                    # ============================== #
                    # 以下有两种计算 ops 的方法
                    # 第一种不统计权重拆分成多个阵列后 mapping 之间的加法
                    # 第二种统计，但是在进行 tops/w 的计算时，有可能超出峰值算力
                    # ============================== #
                    # 方法1 （最终结果不会超过峰值算力）：
                    layer_ops = layer_ops - (math.ceil(rows / size_dmac[0]) - 1) * cols * mvm_times
                    tot_ops_dmac += layer_ops
                    # 方法2：
                    # tot_ops_dmac += layer_ops
                    # ============================== #
                    layer_topsw = layer_ops / (energy_dmac_per_cal * dmac_cal_times) / 1e12
                    assert layer_topsw <= tops_j_dmac
                    # ============================== #
                    layer_dict[name] = {}
                    layer_dict[name]['device'] = 'DMAC'
                    layer_dict[name]['energy'] = layer_energy
                    layer_dict[name]['splits'] = math.ceil(rows / size_dmac[0]) * math.ceil(cols / size_dmac[1])
                    layer_dict[name]['ops'] = layer_ops
                    layer_dict[name]['tops/w'] = layer_topsw

                    if hasattr(module, 'layer_flag'):
                        layer_flag = module.layer_flag
                        if layer_flag in reg_dict.digital_compute_layers:
                            tot_energy_adapter += layer_energy
                            # ============================== #
                            # 以下有两种计算 ops 的方法
                            # 第一种不统计权重拆分成多个阵列后 mapping 之间的加法
                            # 第二种统计，但是在进行 tops/w 的计算时，有可能超出峰值算力
                            # ============================== #
                            # 方法1 （最终结果不会超过峰值算力）：
                            layer_ops = layer_ops - (math.ceil(rows / size_dmac[0]) - 1) * cols
                            tot_ops_adapter += layer_ops
                            # 方法2：
                            # tot_ops_adapter += layer_ops
                            # ============================== #

        ret_dict = {}
        ret_dict['config'] = {}
        ret_dict['config']['tops_j_dmac'] = tops_j_dmac
        ret_dict['config']['tops_j_acim'] = tops_j_acim
        ret_dict['config']['size_acim'] = size_acim
        ret_dict['config']['size_dmac'] = size_dmac

        ret_dict['energy_acim'] = tot_energy_acim
        ret_dict['energy_dmac'] = tot_energy_dmac
        ret_dict['energy_adapter'] = tot_energy_adapter
        ret_dict['energy_adapter (%)'] = tot_energy_adapter / (tot_energy_acim + tot_energy_dmac)
        ret_dict['energy_tot'] = tot_energy_acim + tot_energy_dmac
        ret_dict['ops_acim'] = tot_ops_acim
        ret_dict['ops_dmac'] = tot_ops_dmac
        ret_dict['ops_adapter'] = tot_ops_adapter
        ret_dict['ops_tot'] = tot_ops_acim + tot_ops_dmac

        ret_dict['tops/w (Total)'] = ret_dict['ops_tot'] / 1e12 / ret_dict['energy_tot']
        if ret_dict['energy_acim'] != 0:
            ret_dict['tops/w (ACIM)'] = ret_dict['ops_acim'] / 1e12 / ret_dict['energy_acim']
        if ret_dict['energy_dmac'] != 0:
            ret_dict['tops/w (DMAC)'] = ret_dict['ops_dmac'] / 1e12 / ret_dict['energy_dmac']
        return ret_dict, layer_dict

    def gen_ops_dict(self, input_data):
        ops_dict_all = {}
        model_temp = copy.deepcopy(self)
        _ = model_temp.forward_with_hooks(input_data)  # 前向传播以捕获每层的输入特征图
        self.layer_input_shape = model_temp.layer_input_shape
        for name, module in self.model.named_modules():
            if name in self.layer_input_shape and type(module) in reg_dict.op_layers:
                input_size = self.layer_input_shape[name]
                ops_dict = self.cal_ops(module, input_size)
                module.ops_dict = ops_dict
                ops_dict_all[name] = ops_dict
        return ops_dict_all

    def est_cal_time(self, input_data):
        ops_dict = self.gen_ops_dict(input_data)
        mvm_time_tot = 0
        for name, module in self.model.named_modules():
            # Check if the current module is a convolutional or linear layer, and replace accordingly
            if name in ops_dict:
                mvm_times = ops_dict[name]['mvm_times']
                if type(module) in reg_dict.cim_layers:
                    for key, weight_info in module.weight_mapping_info.items():
                        col_num = weight_info['col_num']
                        if hasattr(module, 'adc_gain'):
                            it_time = module.adc_gain.data.item()
                        else:
                            print(f'No ADC gain info !! Using gain = 2 for time estimation.')
                            it_time = 2
                        mvm_time_weight_block = mvm_time_est_144k(col_num, round(it_time)) * mvm_times
                        mvm_time_tot += mvm_time_weight_block
        print(f'total ACIM mvm time = {mvm_time_tot:.3g}s')
        return mvm_time_tot

    def forward_with_hooks(self, input_data):
        """
        前向传播并记录每层的输入特征图
        """
        self.layer_input_shape = {}
        hooks = []
        model_temp = copy.deepcopy(self.model)

        def hook_factory(name):
            def hook(module, input, output):
                self.layer_input_shape[name] = input[0].shape

            return hook

        for name, module in model_temp.named_modules():
            hook = hook_factory(name)
            hooks.append(module.register_forward_hook(hook))

        output = model_temp(input_data)

        for hook in hooks:
            hook.remove()

        return output

    def get_total_ops(self, input_data):
        ops_dict = self.gen_ops_dict(input_data)

        total_ops = 0
        enhance_ops = 0
        regular_ops = 0

        for name, module in self.model.named_modules():
            if name in ops_dict and type(module) in reg_dict.op_layers:
                ops = ops_dict[name]['total_ops']
                total_ops += ops
                if hasattr(module, 'layer_flag'):
                    layer_flag = module.layer_flag
                else:
                    layer_flag = 'regular'
                if layer_flag in reg_dict.digital_compute_layers:
                    enhance_ops += ops
                else:
                    regular_ops += ops

        return {
            'total_ops': total_ops,
            'enhance_ops': enhance_ops,
            'regular_ops': regular_ops,
            'enhance %': enhance_ops / total_ops * 100,
        }

    def plot_total_ops(self, input_data):
        """
        根据各层的 total_ops 绘制柱状图
        参数:
        ops_dict (dict): 包含各层操作数信息的字典
        """
        ops_dict = self.gen_ops_dict(input_data)
        layer_names = []
        total_ops = []
        colors = []

        for name, module in self.model.named_modules():
            if name in ops_dict:
                ops = ops_dict[name]['total_ops']
                if hasattr(module, 'layer_flag'):
                    layer_flag = module.layer_flag
                else:
                    layer_flag = 'regular'
                layer_names.append(name)
                total_ops.append(ops)
                if layer_flag in reg_dict.digital_compute_layers:
                    colors.append('lightblue')  # 浅蓝色
                else:
                    colors.append('lightgreen')  # 浅绿色

        # 反转列表以倒序排列纵坐标
        layer_names = layer_names[::-1]
        total_ops = total_ops[::-1]
        colors = colors[::-1]

        plt.figure(figsize = (10, 6))
        plt.barh(range(len(total_ops)), total_ops, color = colors, tick_label = layer_names)
        plt.ylabel('Layer Name')
        plt.xlabel('Total Ops')
        plt.title('Total Ops per Layer')
        plt.tight_layout()
        plt.show()

    def update_self_parameter(self,
                              param_dict
                              ):
        for key, value in param_dict.items():
            if key in self.para_list and value is not None:
                setattr(self, key, value)

    def update_layer_parameter(self,
                               update_layer_type_list,
                               **kwargs):
        update_flag = 0
        self.update_self_parameter(kwargs)
        # Check if the provided layer_type exists in custom_conv_layers or custom_linear_layers
        tar_classes = []

        for cls in reg_dict.op_layers:
            for layer_type in update_layer_type_list:
                # print(layer_type)
                # print(cls.__module__.split('.'))
                if layer_type in cls.__module__.split('.'):
                    tar_classes.append(cls)
        # print(f'tar_classes = {tar_classes}')

        for module in self.model.modules():
            if type(module) in tar_classes:
                module.update_para(**kwargs)
                update_flag += 1

        if update_flag == 0:
            print(f'No Layer Params Updated. Program Ended.')
            exit(1)

    def get_qn_parameter(self):
        para_dict = {}
        for param in self.para_list:
            if hasattr(self, param):  # Check if the attribute exists in self
                para_dict[param] = getattr(self, param)  # Get the attribute value
        return para_dict

    def find_and_replace_module(self, parent, name, new_module):
        """递归查找并替换指定名称的模块"""
        attrs = name.split('.')
        for i, attr in enumerate(attrs):
            if i == len(attrs) - 1:
                # 最后一级，执行替换
                setattr(parent, attr, new_module)
            else:
                # 非最后一级，继续递归
                parent = getattr(parent, attr)

    def get_onnx_layers(self, onnx_model):
        name_dict = {}
        # 遍历模型中的所有模块，收集需要替换的模块信息
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_layers:
                onnx_layer_name = hbt.get_onnx_layer_name(module, onnx_model)
                if onnx_layer_name is None:
                    raise ValueError(f"Corresponding ONNX layer not found. Name = {name}")
                name_dict[name] = onnx_layer_name
        return name_dict

    # Assume custom_linear_layers and custom_conv_layers are imported or defined elsewhere
    def convert_to_layers(self,
                          convert_layer_type_list,
                          tar_layer_type,
                          exclude_layers = None,
                          assign_layers = None,
                          **kwargs):
        # Assign any parameter from kwargs to self if it exists in self.para_list
        for key, value in kwargs.items():
            if key in self.para_list:
                setattr(self, key, value)  # Dynamically set self.<parameter_name> = value

        # Check if the provided layer_type exists in custom_conv_layers or custom_linear_layers
        tar_conv_class = None
        for cls in reg_dict.conv_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_conv_class = cls
                break

        # Determine the linear class based on layer_type
        tar_linear_class = None
        for cls in reg_dict.linear_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_linear_class = cls
                break

        if tar_conv_class is None and tar_linear_class is None:
            raise ValueError(f"Invalid layer_type '{tar_layer_type}'. Ensure it is correct and registered.")

        # Ensure not both exclude_layers and assign_layers are provided
        if exclude_layers is not None and assign_layers is not None:
            raise ValueError("Either 'exclude_layers' or 'assign_layers' should be provided, but not both.")

        to_replace = []
        for name, module in self.model.named_modules():
            if exclude_layers and name in exclude_layers:
                continue
            if assign_layers and name not in assign_layers:
                continue
            if type(module) in convert_layer_type_list:
                dev = next(module.parameters()).device
                print(f'original moudle is on device: {dev}')
                if 'conv' in type(module).__name__.lower():
                    new_module = tar_conv_class(
                        in_channels = module.in_channels,
                        out_channels = module.out_channels,
                        kernel_size = module.kernel_size,
                        stride = module.stride,
                        padding = module.padding,
                        groups = module.groups,
                        bias = (module.bias is not None),
                        **kwargs
                    )

                elif 'linear' in type(module).__name__.lower():
                    new_module = tar_linear_class(
                        in_features = module.in_features,
                        out_features = module.out_features,
                        bias = (module.bias is not None),
                        **kwargs
                    )
                else:
                    raise NotImplementedError
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                self.copy_meta_info(module, new_module)
                self.copy_lsq_data(module, new_module)
                to_replace.append((name, new_module))

        print(f"\n=============================================")
        if not to_replace:
            print(f"No layers converted to {tar_layer_type} layers.")
        for name, new_module in to_replace:
            print(f"Converted to {tar_layer_type} Layer: {name}")
            self.find_and_replace_module(self.model, name, new_module)
        print(f"=============================================\n")
        # self.set_device()

    def convert_to_modules(self,
                           convert_layer_type_list,
                           tar_layer_type,
                           exclude_layers = None,
                           assign_layers = None,
                           **kwargs):
        # Assign any parameter from kwargs to self if it exists in self.para_list
        for key, value in kwargs.items():
            if key in self.para_list:
                setattr(self, key, value)  # Dynamically set self.<parameter_name> = value

        # Check if the provided layer_type exists in custom_conv_layers or custom_linear_layers
        tar_conv_class = None
        for cls in reg_dict.conv_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_conv_class = cls
                break

        # Determine the linear class based on layer_type
        tar_linear_class = None
        for cls in reg_dict.linear_layers:
            if tar_layer_type in cls.__module__.split('.'):
                tar_linear_class = cls
                break

        if tar_conv_class is None and tar_linear_class is None:
            raise ValueError(f"Invalid layer_type '{tar_layer_type}'. Ensure it is correct and registered.")

        # Ensure not both exclude_layers and assign_layers are provided
        if exclude_layers is not None and assign_layers is not None:
            raise ValueError("Either 'exclude_layers' or 'assign_layers' should be provided, but not both.")

        to_replace = []
        for name, module in self.model.named_modules():
            if exclude_layers and name in exclude_layers:
                continue
            if assign_layers and name not in assign_layers:
                continue
            if type(module) in convert_layer_type_list:
                if 'conv' in type(module).__name__.lower():
                    new_module = tar_conv_class(module, **kwargs)

                elif 'linear' in type(module).__name__.lower():
                    new_module = tar_linear_class(module, **kwargs)
                else:
                    raise NotImplementedError
                to_replace.append((name, new_module))

        print(f"\n=============================================")
        if not to_replace:
            print(f"No layers converted to {tar_layer_type} layers.")
        for name, new_module in to_replace:
            print(f"Converted to {tar_layer_type} Layer: {name}")
            self.find_and_replace_module(self.model, name, new_module)
        print(f"=============================================\n")
        # self.set_device()

    def convert_to_lsq_int_layers(self,
                                  int_grad = False,
                                  x_detach = False,
                                  weight_bit_extension = 4,
                                  exclude_layers = None,
                                  assign_layers = None,
                                  train_lsq_int_layers = False,
                                  ):
        to_replace = []
        for name, module in self.model.named_modules():
            if exclude_layers and name in exclude_layers:
                continue
            if assign_layers and name not in assign_layers:
                continue

            if type(module) in reg_dict.qn_layers:
                if 'conv' in type(module).__name__.lower():
                    new_module = l_lsq_int.Conv2d_lsq_int(module,
                                                          x_detach = x_detach,
                                                          int_grad = int_grad,
                                                          weight_bit_extension = weight_bit_extension)

                elif 'linear' in type(module).__name__.lower():
                    new_module = l_lsq_int.Linear_lsq_int(module,
                                                          int_grad = int_grad,
                                                          weight_bit_extension = weight_bit_extension)
                else:
                    raise NotImplementedError
                self.copy_meta_info(module, new_module)
                self.copy_lsq_data(module, new_module)
                # ================== #
                # LSQ_int 层自动拟合训练
                # ================== #
                # 效果不好，不要使用！！
                if train_lsq_int_layers:
                    self.train_lsq_int_layers(lsq_net = copy.deepcopy(module), lsq_int_net = new_module)
                to_replace.append((name, new_module))

        print(f"\n=============================================")
        if not to_replace:
            print(f"No layers converted to INT LSQ layers.")
        for name, new_module in to_replace:
            print(f"Converted to INT LSQ Layer: {name}")
            self.find_and_replace_module(self.model, name, new_module)
        print(f"=============================================\n")

    def train_lsq_int_layers(self, lsq_net, lsq_int_net, batch_size = 10):
        if hasattr(lsq_net, 'in_channels'):
            x_shape = (batch_size, lsq_net.in_channels, 50, 50)
        else:
            x_shape = (batch_size, lsq_net.in_features)

        # ================ #
        # 训练前散点图
        # ================ #
        # x = torch.rand(x_shape, device = self.device)
        # lsq_int_out = lsq_int_net(x)
        # lsq_out = lsq_net(x)
        # scatter_plt(lsq_int_out, lsq_out, title = 'Before Training')

        # ================ #
        # 训练超参数
        # ================ #
        lr = 1e-3
        min_lr = 1e-8
        patience = 100
        factor = 0.1
        epochs = 1000

        optimizer = torch.optim.Adam(lsq_int_net.parameters(), lr = lr)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = factor,
                                                               patience = patience, verbose = True, min_lr = min_lr)

        loss_history = []
        lsq_net.eval()
        for epoch in range(epochs):
            x = torch.rand(x_shape, device = self.device)

            lsq_int_out = lsq_int_net(x)
            lsq_out = lsq_net(x)
            loss = torch.sum((lsq_out - lsq_int_out) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedular.step(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
            loss_history.append(loss.item())

        # ================ #
        # 训练后散点图和loss图
        # ================ #
        # scatter_plt(lsq_int_out, lsq_out, title = 'After Training')
        # plt.plot(range(1, epochs + 1), loss_history, marker = 'o')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title(f'Training Loss for {lsq_net.name}')
        # plt.show()

    def revert_to_nn_layer(self,
                           model = None,
                           exclude_layers = None,
                           assign_layers = None,
                           verbose = True):
        # 初始化一个空列表来存储待更新的模块信息（名称和标准模块）
        to_replace = []
        if model is None:
            model = self.model

        # 遍历模型中的所有模块，收集需要替换的模块信息
        for name, module in model.named_modules():
            if exclude_layers is not None:
                # Skip layers listed in excluded_layers
                if name in exclude_layers:
                    continue
            elif assign_layers is not None:
                # Only include layers listed in assign_layers
                if name not in assign_layers:
                    continue

            if type(module) in reg_dict.custom_conv_layers:
                # 创建对应的标准卷积层，但不立即替换
                new_module = nn.Conv2d(
                    in_channels = module.in_channels,
                    out_channels = module.out_channels,
                    kernel_size = module.kernel_size,
                    stride = module.stride,
                    groups = module.groups,
                    padding = module.padding,
                    bias = (module.bias is not None),
                )
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                self.copy_meta_info(module, new_module)
                to_replace.append((name, new_module))

            elif type(module) in reg_dict.custom_linear_layers:
                # 创建对应的标准全连接层，但不立即替换
                new_module = nn.Linear(
                    in_features = module.in_features,
                    out_features = module.out_features,
                    bias = (module.bias is not None),
                )
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                self.copy_meta_info(module, new_module)
                to_replace.append((name, new_module))

        if verbose:
            print(f'\n=============================================')
            if len(to_replace) == 0:
                print(f'No Layer Reverted to NN Layer')

            for name, new_module in to_replace:
                print(f'Reverted to NN Layer: {name}')
                self.find_and_replace_module(model, name, new_module)
            print(f'=============================================\n')

        # self.set_device()

    def copy_meta_info(self, module, new_module):
        new_module.name = module.name
        if hasattr(module, 'layer_flag'):
            new_module.layer_flag = getattr(module, 'layer_flag')

    def copy_lsq_data(self, module, new_module):
        if hasattr(module, 'step_size_weight') and hasattr(new_module, 'step_size_weight'):
            new_module.step_size_weight.data = module.step_size_weight.data
            new_module.step_size_input.data = module.step_size_input.data
            new_module.step_size_output.data = module.step_size_output.data

    def add_enhance_layers(self, ops_factor = 0.05):
        to_replace = []

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_conv_layers:
                # 在 Conv2d_qn_lsq 层后添加 EnhanceLayerConv2d
                new_module = en.EnhanceLayerConv2d(module, ops_factor = ops_factor)
                to_replace.append((name, new_module))
            elif type(module) in reg_dict.custom_linear_layers:
                # 在 Linear_qn_lsq 层后添加 EnhanceLayerLinear
                new_module = en.EnhanceLayerLinear(module, ops_factor = ops_factor)
                to_replace.append((name, new_module))

        for name, new_module in to_replace:
            self.find_and_replace_module(self.model, name, new_module)

        # self.set_device()
        self.assign_module_name()

    # def add_enhance_branch(self, ops_factor = 0.05):
    #     to_replace = []
    #
    #     for name, module in self.model.named_modules():
    #         if type(module) in reg_dict.custom_conv_layers:
    #             new_module = en.EnhanceBranchConv2d_LoR(
    #                 original_conv = module,
    #                 ops_factor = ops_factor
    #             )
    #             to_replace.append((name, new_module))
    #
    #         elif type(module) in reg_dict.custom_linear_layers:
    #             new_module = en.EnhanceBranchLinear_LoR(original_linear = module,
    #                                                     ops_factor = ops_factor,
    #                                                     )
    #             to_replace.append((name, new_module))
    #
    #     for name, new_module in to_replace:
    #         self.find_and_replace_module(self.model, name, new_module)
    #
    #     self.set_device()
    #     self.assign_module_name()

    def add_enhance_branch_LoR(self,
                               ops_factor = 0.05,
                               relu = False,
                               sigmoid = True):
        to_replace = []

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.custom_conv_layers:
                new_module = en.EnhanceBranchConv2d_LoR(original_conv = module,
                                                        relu = relu,
                                                        sigmoid = sigmoid,
                                                        ops_factor = ops_factor
                                                        )
                to_replace.append((name, new_module))

            elif type(module) in reg_dict.custom_linear_layers:
                new_module = en.EnhanceBranchLinear_LoR(original_linear = module,
                                                        relu = relu,
                                                        sigmoid = sigmoid,
                                                        ops_factor = ops_factor
                                                        )
                to_replace.append((name, new_module))

        for name, new_module in to_replace:
            self.find_and_replace_module(self.model, name, new_module)

        # self.set_device()
        self.assign_module_name()

    def zero_qn_layers(self):
        for name, module in self.model.named_modules():
            if type(module) in (l_qn_lsq.Conv2d_qn_lsq, l_qn_lsq.Linear_qn_lsq):
                for p_name, param in module.named_parameters():
                    param.detach()
                    param.data = param.data * 0

    def zero_branch_layers(self):
        for name, module in self.model.named_modules():
            if getattr(module, 'layer_flag', None) == 'enhance_branch':
                for p_name, param in module.named_parameters():
                    param.detach()
                    param.data = param.data * 0

    def set_blend_factors(self, value = 0.5):
        # 计算需要填充的数值
        logit_value = torch.log(torch.tensor(value) / (1 - torch.tensor(value)))
        for name, module in self.model.named_modules():
            if getattr(module, 'blend_factor', None) is not None:
                # 填充计算得到的logit值
                module.blend_factor.data.fill_(logit_value.item())

    def get_blend_factors(self):
        blend_factor_dict = {}
        blend_values = []
        blend_intensities = []
        # Iterate through the modules to collect blend factors
        for name, module in self.model.named_modules():
            if getattr(module, 'blend_factor', None) is not None:
                # Collect the blend factor data
                blend_factor_data = module.blend_factor.data
                blend_value = torch.sigmoid(blend_factor_data)
                blend_factor_dict[name] = blend_value
                # Append the blend factor to the list for mean calculation
                blend_values.append(blend_value)

                std_ori = module.std_ori
                mean_ori = module.mean_ori
                std_enhance = module.std_enhance
                mean_enhance = module.mean_enhance

                blend_intensity = std_ori * (1 - blend_value) / std_enhance * blend_value
                blend_intensities.append(blend_intensity)
                # Calculate the mean of all blend factors
        if blend_values:
            blend_value_mean = torch.mean(torch.stack(blend_values))
            blend_intensity_mean = torch.mean(torch.stack(blend_intensities))
        else:
            blend_value_mean = None  # Handle the case where there are no blend factors
            blend_intensity_mean = None
        return blend_factor_dict, blend_value_mean, blend_intensity_mean

    def set_requires_grad(self, name, p_name, param, requires_grad):
        param.requires_grad = requires_grad
        action = "Froze" if not requires_grad else "Unfroze"
        print(f"{action} parameter: {name}-{p_name}")

    def freeze_adc_gain(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                for p_name, param in module.named_parameters():
                    if p_name == 'adc_gain':
                        self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_step_size(self,
                         freeze_in_s = True,
                         freeze_out_s = True,
                         freeze_w_s = True,
                         requires_grad = False):
        p_name_list = []
        if freeze_in_s:
            p_name_list.append('step_size_input')
        if freeze_out_s:
            p_name_list.append('step_size_output')
        if freeze_w_s:
            p_name_list.append('step_size_weight')
        # p_name_list = ['step_size_input', 'step_size_output','step_size_weight']

        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                for p_name, param in module.named_parameters():
                    if p_name in p_name_list:
                        self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_adda_layers(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_qn_layers(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if type(module) in (l_qn_lsq.Conv2d_qn_lsq, l_qn_lsq.Linear_qn_lsq):
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_blend_factors(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if getattr(module, 'blend_factor', None) is not None:
                self.set_requires_grad(name, 'blend_factor', module.blend_factor, requires_grad)

    def freeze_bn_layers(self, requires_grad = False):
        """
        遍历给定的 PyTorch 模型，冻结所有 BatchNorm 层的权重，并将这些层设置为 eval 模式，
        确保训练时它们使用 eval 模式下的参数。

        Args:
            model: 需要处理的 PyTorch 模型。
        """
        for layer in self.model.modules():
            # 检查是否是 BatchNorm 层
            if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
                # 冻结权重和偏置参数
                layer.eval()  # 设置为 eval 模式，使用推理中的统计数据
                for param in layer.parameters():
                    param.requires_grad = requires_grad  # 冻结权重，不参与训练

    def freeze_enhance_branch(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if getattr(module, 'layer_flag', None) == 'enhance_branch':
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    def freeze_enhance_layer(self, requires_grad = False):
        for name, module in self.model.named_modules():
            if getattr(module, 'layer_flag', None) == 'enhance_layer':
                for p_name, param in module.named_parameters():
                    self.set_requires_grad(name, p_name, param, requires_grad)

    @staticmethod
    def get_step(d_range, cycles):
        step = (d_range[1] - d_range[0]) / (cycles - 1) if cycles > 1 else 0
        return step

    @staticmethod
    def get_step_exp(d_range, cycles):
        if cycles < 2:
            return [0]

        # 创建一个从1到10的对数线性序列
        scale = np.linspace(2, 1, cycles - 1)
        scale = np.exp(scale - 1)  # 对数变换后前面增长较快

        # 归一化后按比例计算每个步长
        normalized_scale = scale / scale.sum()
        step_list = normalized_scale * (d_range[1] - d_range[0])

        return step_list.tolist()

    def compare_model_weights(self, model1, model2):
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()

        same_weights = []
        different_weights = []

        # Compare the weights
        for key in model1_state_dict.keys():
            if key in model2_state_dict:
                if torch.equal(model1_state_dict[key], model2_state_dict[key]):
                    same_weights.append(key)
                else:
                    different_weights.append(key)
            else:
                different_weights.append(key)

        # Add weights that are in model2 but not in model1
        for key in model2_state_dict.keys():
            if key not in model1_state_dict:
                different_weights.append(key)

        # Print the results
        print(f'----------------')
        print("Same Weights:")
        print(f'----------------')
        for key in same_weights:
            print(key)
        print(f'\n')
        print(f'----------------')
        print("Different Weights:")
        print(f'----------------')
        for key in different_weights:
            print(key)

        return {
            "same_weights": same_weights,
            "different_weights": different_weights
        }

    def train_enhance_layer_w_teacher(self, training_dataset,
                                      teacher_model,
                                      path,
                                      patience = 2,
                                      cooldown = 0,
                                      factor = 0.5,
                                      epoch = 1,
                                      lr = 1e-3):
        # 初始化优化器，仅优化 enhance_layer 的权重和 bias
        # 遍历模型中的所有模块
        min_lr = lr / 100
        enhance_parameters = []
        enhance_parameters_name = []
        for name, module in self.model.named_modules():
            # 检查模块是否有 'layer_flag' 属性且其值为 'enhance_layer'
            if getattr(module, 'layer_flag', None) in reg_dict.digital_compute_layers:
                # 遍历模块的所有参数
                for p_name, param in module.named_parameters():
                    # 仅包含 requires_grad 为 True 的参数
                    if param.requires_grad:
                        enhance_parameters.append(param)
                        enhance_parameters_name.append(f'{name}-{p_name}')
        print(f'\n')
        print(f'-----------------------------------')
        print(f'Train Parameters with Teacher Model')
        print(f'-----------------------------------')
        for p_name in enhance_parameters_name:
            print(p_name)

        # print(f'\n')
        # print(f'-----------------------------------')
        # self.compare_model_weights(teacher_model, self.model)
        # print(f'-----------------------------------')
        # print(f'\n')

        optimizer = optim.Adam(enhance_parameters, lr = lr)
        loss_func = nn.MSELoss()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode = 'min',
                                                   factor = factor,
                                                   patience = patience,
                                                   cooldown = cooldown,
                                                   min_lr = min_lr)

        epoch_loss_list = []
        batch_loss_list = []
        # 训练循环
        for e in range(epoch):
            loop = tqdm(training_dataset, leave = True)
            epoch_loss = 0
            for batch_idx, (training_data, _) in enumerate(loop):
                optimizer.zero_grad()

                training_data = training_data.to(self.device)
                # 获取教师模型和学生模型的输出
                _, teacher_output_dict = self.forward_with_hooks_layer_flag(teacher_model, training_data, layer_flag = reg_dict.digital_compute_layers)
                _, student_output_dict = self.forward_with_hooks_layer_flag(self.model, training_data, layer_flag = reg_dict.digital_compute_layers)

                # 初始化总损失
                total_loss = torch.tensor(0.0, device = training_data.device)

                # 计算每个增强层的损失并累加
                for layer_name, feature_map_t in teacher_output_dict.items():
                    feature_map_s = student_output_dict[layer_name]
                    loss = loss_func(feature_map_s, feature_map_t)
                    total_loss += loss

                epoch_loss += total_loss.item()
                batch_loss_list.append(total_loss.item())
                # 反向传播和优化
                total_loss.backward()
                optimizer.step()

                # 更新进度条描述
                loop.set_description(f"Epoch {e + 1}/{epoch} Batch {batch_idx + 1}/{len(training_dataset)}, Loss: {total_loss.item():.4f}")
                save_to_json(batch_loss_list, filename = f'{path}/local_train_batch_loss_list.json')

            lr_old = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss)
            # scheduler.step(epoch_loss)
            lr_current = optimizer.param_groups[0]['lr']

            if lr_current < lr_old:
                print(f'\n')
                print(f'---------------------------------------')
                print(f'LR changed from {lr_old:.3g} to {lr_current:.3g}')
                print(f'---------------------------------------')
                print(f'\n')
            epoch_loss_list.append(epoch_loss)
            save_to_json(epoch_loss_list, filename = f'{path}/local_train_epoch_loss_list.json')

            if lr_current == min_lr:
                break

    # 定义如何获取每层的输出特征图
    # def forward_with_hooks_layer_flag(self, model, input_data, layer_flag = ['enhance_layer']):
    #     output_dict = {}
    #     hooks = []
    #
    #     def hook(module, input, output, name):
    #         if getattr(module, 'layer_flag', None) in layer_flag:
    #             output_dict[name] = output
    #
    #     for name, module in model.named_modules():
    #         hooks.append(module.register_forward_hook(lambda module, input, output, name = name: hook(module, input, output, name)))
    #
    #     output = model(input_data)
    #
    #     for hook in hooks:
    #         hook.remove()
    #
    #     return output, output_dict

    def forward_with_hooks_layer_flag(self, model, input_data, layer_flag = ['enhance_layer']):
        output_dict = {}
        hooks = []

        def hook_factory(name):
            def hook(module, input, output):
                if getattr(module, 'layer_flag', None) in layer_flag:
                    output_dict[name] = output

            return hook

        for name, module in model.named_modules():
            hook = hook_factory(name)
            hooks.append(module.register_forward_hook(hook))

        output = model(input_data)

        for hook in hooks:
            hook.remove()

        return output, output_dict

    def get_adc_config(self):
        adc_config_dict = {}
        adc_adjust_mode = 'gain'
        for name, module in self.model.named_modules():
            if hasattr(module, 'adc_gain'):
                adc_config_dict[name] = {}
                adc_gain = torch.clamp(module.adc_gain.data,
                                       min = module.adc_gain_min,
                                       max = module.adc_gain_max)
                if hasattr(module, 'adc_adjust_mode'):
                    adc_adjust_mode = module.adc_adjust_mode
                if adc_adjust_mode == 'gain':
                    adc_config_dict[name]['gain_level'] = adc_gain.round().item()
                else:
                    adc_config_dict[name]['current_range'] = (1 / adc_gain).round().item()
        return adc_config_dict

    def get_adda_adc_gain_dict(self):
        adc_gain_dict = {}
        for name, module in self.model.named_modules():
            if type(module) in reg_dict.adda_layers:
                adc_gain_module_dict = copy.deepcopy(module.adc_gain_dict)
                for key, val in adc_gain_module_dict.items():
                    adc_gain_module_dict[key] = int(round(val.data.item()))
                adc_gain_dict[name] = adc_gain_module_dict
        return adc_gain_dict

    def progressive_train(self,
                          qn_cycle,
                          update_layer_type_list,
                          start_cycle,
                          **kwargs):
        # Dictionary to store step values
        steps_dict = {}
        current_para_dict = {}
        # Iterate over all arguments, checking for _range in keys
        for param_name, param_value in kwargs.items():
            if param_name.endswith('_range'):
                param = param_name.replace('_range', '')
                # Calculate step using self.get_step and store in steps_dict
                steps_dict[param] = self.get_step(param_value, qn_cycle)
                current_para_dict[param] = param_value[0]

        # Get the signature of train_model and retrieve its parameter names
        train_model_signature = inspect.signature(self.train_model)
        train_model_params = set(train_model_signature.parameters.keys())
        # Filter kwargs to retain only those needed for train_model
        train_model_kwargs = {k: v for k, v in kwargs.items() if k in train_model_params}

        for cyc in range(qn_cycle):
            # Skip cycles until start_cycle is reached
            if cyc < start_cycle:
                for param, step in steps_dict.items():
                    current_para_dict[param] += step  # Update current values
                continue

            # Round the values as needed for certain parameters (e.g., integer bit values)
            rounded_params = {
                key: round(value) if key != 'noise_scale' else value
                for key, value in current_para_dict.items()
            }

            print(f'\n')
            print(f'==============================================')
            print(f'Progressive Training')
            print(f'Layer Type = {update_layer_type_list}')
            print(f'Parameters:')
            for key, value in rounded_params.items():
                if key != 'noise_scale':
                    print(f'{key} = {value}')
                else:
                    print(f'{key} = {value:.3g}')
            print(f'==============================================')
            print(f'\n')
            self.update_layer_parameter(
                update_layer_type_list = update_layer_type_list,
                **rounded_params,
            )

            # Increment each parameter in current_para_dict by its respective step
            for param, step in steps_dict.items():
                current_para_dict[param] += step

            self.train_model(**train_model_kwargs)

    def set_device(self, device = None, device_ids = None):
        if device is None:
            device = self.device
        if device_ids is None:
            device_ids = self.device_ids
        self.device = device
        self.device_ids = device_ids
        self.model.to(device)
        print(f'set model to device: {device}')
        # if device_ids is not None:
        #     self.model = nn.DataParallel(self.model, device_ids = device_ids)
        #     print(f'model.device_ids =  {self.model.device_ids}')

    def load_model(self, PATH, strict = True):
        checkpoint = torch.load(PATH, map_location = self.device)
        model = self.model

        # Handle DDP
        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        real_model = model.module if is_ddp else model

        # Check whether keys have 'module.' prefix
        checkpoint_keys = list(checkpoint.keys())
        if all(k.startswith("module.") for k in checkpoint_keys):
            # checkpoint has 'module.' but model is not DDP-wrapped
            if not is_ddp:
                # remove 'module.' prefix
                checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        elif not all(k.startswith("module.") for k in checkpoint_keys):
            # checkpoint has no 'module.' but model is DDP-wrapped
            if is_ddp:
                # add 'module.' prefix
                checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}

        real_model.load_state_dict(checkpoint, strict = strict)
        print("✅ Model loaded!")

    def load_model(self, PATH, strict = True):
        checkpoint = torch.load(PATH, map_location = self.device)
        model = self.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model.load_state_dict(checkpoint, strict = strict)
        print("✅ Model loaded!")

    def save_model(self, PATH):
        # 获取目录路径
        dir_path = os.path.dirname(PATH)
        if not os.path.exists(dir_path) and len(dir_path) > 0:
            os.makedirs(dir_path)

        # 处理 DDP 包裹的模型
        model_to_save = self.model
        if isinstance(model_to_save, torch.nn.parallel.DistributedDataParallel):
            model_to_save = model_to_save.module

        # 保存 state_dict
        torch.save(model_to_save.state_dict(), PATH)
        if hasattr(self, 'rank'):
            if self.rank == 0:
                print(f'✅ Model saved at {PATH}')
        else:
            print(f'✅ Model saved at {PATH}')

    def remove_module_prefix(self, state_dict):
        """移除所有键中的'module.'前缀"""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key  # 去除'module.'前缀
            new_state_dict[new_key] = value
        return new_state_dict

    def add_excluded_layers(self, excluded_layers):
        self.excluded_layers = excluded_layers

    def gen_model_name(self, str_suffix = None):
        model_name = f'{self.model_name}_w={self.weight_bit}b_i={self.input_bit}b_o={self.output_bit}b_n={self.noise_scale:.3g}'
        if str_suffix is not None:
            model_name = f'{model_name}_{str_suffix}'
        return model_name

    def plot_loss(self, loss_list, title = 'Training Loss', save_path = None):
        if save_path is None:
            save_path = self.gen_model_name('_loss')
            save_path = f'{save_path}.png'
        # 获取目录路径
        dir_path = os.path.dirname(save_path)
        # 检查目录是否存在，如果不存在，则创建它
        if not os.path.exists(dir_path) and len(dir_path) > 0:
            os.makedirs(dir_path)
        plt.figure()  # 创建一个新的图形
        plt.plot(loss_list)
        plt.title(title)
        plt.savefig(save_path)  # 保存图形到指定路径
        plt.close()  # 关闭图形，释放内存

    def export_onnx(self, input_data, onnx_path = None):
        # ======================================== #
        # 删除 self.model 当中的运行时创建的 Non-Leaf Tensor
        # ======================================== #
        def find_non_leaf_tensors(module):
            bad_attrs = []

            def _check(obj, prefix = ""):
                if isinstance(obj, torch.Tensor):
                    if not obj.is_leaf:
                        bad_attrs.append(prefix)
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        _check(v, f"{prefix}.{k}" if prefix else k)
                elif isinstance(obj, (list, tuple)):
                    for idx, item in enumerate(obj):
                        _check(item, f"{prefix}[{idx}]")
                elif hasattr(obj, '__dict__'):
                    for attr_name in dir(obj):
                        if attr_name.startswith("__"):
                            continue
                        try:
                            val = getattr(obj, attr_name)
                            _check(val, f"{prefix}.{attr_name}" if prefix else attr_name)
                        except Exception:
                            pass

            _check(module)
            return bad_attrs

        def make_leaf_tensor(obj, attr_path):
            parts = attr_path.split('.')
            current = obj
            for i, p in enumerate(parts[:-1]):
                if isinstance(current, dict):  # 用 key 访问
                    if p in current:
                        current = current[p]
                    else:
                        raise AttributeError(f"Key '{p}' not found in dict at {'.'.join(parts[:i])}")
                else:  # 普通 getattr
                    if hasattr(current, p):
                        current = getattr(current, p)
                    elif hasattr(current, '_modules') and p in current._modules:
                        current = current._modules[p]
                    else:
                        raise AttributeError(f"Attribute '{p}' not found in {current}")

            # 最后一级赋值
            final_attr = parts[-1]
            leaf_tensor = getattr(current, final_attr)
            leaf_tensor = leaf_tensor.detach().clone()
            leaf_tensor.requires_grad_(False)
            setattr(current, final_attr, leaf_tensor)
            # print(f"✔ Fixed: {attr_path}")

        bad_tensors = find_non_leaf_tensors(self.model)
        # print(f'bad_tensors = {bad_tensors}')
        for path in bad_tensors:
            make_leaf_tensor(self.model, path)
        # ======================================== #
        # 删除完毕
        # ======================================== #
        model_ = copy.deepcopy(self.model)
        if onnx_path is None:
            onnx_path = self.gen_model_name()

        # 获取目录路径
        dir_path = os.path.dirname(onnx_path)
        # 检查目录是否存在，如果不存在，则创建它
        if not os.path.exists(dir_path) and len(dir_path) > 0:
            os.makedirs(dir_path)

        self.revert_to_nn_layer(model = model_, verbose = True)
        model_.eval()

        def to_device(data):
            if isinstance(data, list):
                return [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in data]
            elif isinstance(data, tuple):
                return tuple(item.to(self.device) if isinstance(item, torch.Tensor) else item for item in data)
            elif isinstance(data, torch.Tensor):
                return data.to(self.device)
            else:
                return data

        input_data = to_device(input_data)
        model_path = f"{onnx_path}.onnx"
        torch.onnx.export(model_,  # model being run
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

    def plot_model_parameters(self, model):
        blend_factor = {}
        weights_list = []
        biases_list = []

        # 第一次遍历，统计所有层权重和偏置的最大值和最小值
        for name, module in model.named_modules():
            if type(module) in (nn.Conv2d, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                weights_list.append(weights)
                if module.bias is not None:
                    biases = module.bias.data.cpu().numpy()
                    biases_list.append(biases)

        # 确定所有层的权重和偏置的全局最大值和最小值
        all_weights = np.concatenate([w.flatten() for w in weights_list])
        all_biases = np.concatenate([b.flatten() for b in biases_list]) if biases_list else np.array([0])
        global_min = min(all_weights.min(), all_biases.min())
        global_max = max(all_weights.max(), all_biases.max())

        # 打印和绘图
        fig, axs = plt.subplots(len(weights_list), 1, figsize = (12, 6 * len(weights_list)))
        if len(weights_list) == 1:
            axs = [axs]

        i = 0
        for j, (name, module) in enumerate(model.named_modules()):
            if type(module) in (nn.Conv2d, nn.Linear):
                weights = module.weight.data.cpu().numpy()
                biases = module.bias.data.cpu().numpy() if module.bias is not None else None

                # 打印权重和偏置的均值和标准差
                print(f'==============')
                print(f'Name: {name}')
                print(f'Weight Mean: {weights.mean()}')
                print(f'Weight Std: {weights.std()}')
                if biases is not None:
                    print(f'Bias Mean: {biases.mean()}')
                    print(f'Bias Std: {biases.std()}')
                else:
                    print('Biases: None')
                print(f'==============')

                # 绘制权重分布图
                axs[i].hist(weights.flatten(), bins = 50, alpha = 0.7, label = 'Weights', range = (global_min, global_max))
                if biases is not None:
                    axs[i].hist(biases.flatten(), bins = 50, alpha = 0.7, label = 'Biases', range = (global_min, global_max))
                axs[i].set_title(f'Distribution of Weights and Biases for {name}')
                axs[i].set_xlabel('Value')
                axs[i].set_ylabel('Frequency')
                axs[i].legend()
                i += 1
            if getattr(module, 'blend_factor', None) is not None:
                blend_factor[name] = F.sigmoid(module.blend_factor).detach().cpu().numpy()

        plt.tight_layout()
        plt.show()

        for key, value in blend_factor.items():
            print(f'{key}: {value}')
