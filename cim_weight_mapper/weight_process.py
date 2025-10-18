import cim_layers.register_dict as reg_dict
from .weight_mapper import *


def get_2d_weight(weight):
    weight_2d = weight.reshape(weight.shape[0], -1).transpose(0, 1)
    return weight_2d


def gen_weight_split_dict(module, weight_block_size):
    max_rows = weight_block_size[0]
    max_cols = weight_block_size[1]
    weight_2d = get_2d_weight(module.weight)
    rows, cols = weight_2d.shape

    # 权重切分数量
    num_blocks_row = rows // max_rows
    num_blocks_col = cols // max_cols

    module_weight_mapping_info = {}

    for row_block in range(num_blocks_row + 1):
        for col_block in range(num_blocks_col + 1):
            start_row = row_block * max_rows
            start_col = col_block * max_cols
            # 计算实际行和列数量，防止超出矩阵范围
            actual_rows = min(max_rows, rows - start_row)
            actual_cols = min(max_cols, cols - start_col)
            if actual_rows <= 0 or actual_cols <= 0:
                continue

            key = f"{row_block}_{col_block}"
            module_weight_mapping_info[key] = {
                "start_row": start_row,
                "start_col": start_col,
                "row_num": actual_rows,
                "col_num": actual_cols,
            }
    module_weight_mapping_info = module_weight_mapping_info
    return module_weight_mapping_info


def convert_to_cim_weights(model, weight_block_size, module_for_map,
                           assign_layers, exclude_layers):
    if exclude_layers is not None and assign_layers is not None:
        raise ValueError("Either 'excluded_layers' or 'assign_layers' should be provided, but not both.")

    num_layers = 0
    model_weight_mapping_info = {}
    # 遍历所有计算层
    for name, module in model.named_modules():
        # Check if the current module is a convolutional or linear layer, and replace accordingly
        if type(module) in module_for_map:
            if exclude_layers is not None:
                # Skip layers listed in excluded_layers
                if name in exclude_layers:
                    continue
            elif assign_layers is not None:
                # Only include layers listed in assign_layers
                if name not in assign_layers:
                    continue
            # ===================================== #
            # 权重拆分
            # ===================================== #
            weight_mapping_info = gen_weight_split_dict(module = module, weight_block_size = weight_block_size)
            module.weight_mapping_info = weight_mapping_info
            print(f'Generated CIM weight: {name}')
            num_layers += 1
            model_weight_mapping_info[name] = weight_mapping_info
    if num_layers == 0:
        print(f'No CIM layers. No weight converted.')

    return model_weight_mapping_info


def weight_info_to_blocks(weight_info):
    block_dict = {}
    lut = {}
    for layer_key, module_weight_mapping_info in weight_info.items():
        for split_key, mapping_info in module_weight_mapping_info.items():
            # 确保block_key唯一性，可以在block_key前加上layer_key作为前缀
            unique_block_key = f"{layer_key}_{split_key}"
            rows = mapping_info['row_num']
            cols = mapping_info['col_num']
            # 提取weight_data
            weight_data = torch.zeros(rows, cols)
            if weight_data is not None:
                # 将weight_data保存在block_dict中
                block_dict[unique_block_key] = {'data': weight_data}
                # 在LUT中记录完整的索引路径
                lut[unique_block_key] = {}
                lut[unique_block_key]['layer_key'] = layer_key
                lut[unique_block_key]['split_key'] = split_key
    return block_dict, lut


def sort_block_dict(block_dict):
    # 使用sorted函数排序block_dict，按列数从大到小排序，如果列数相同，则按行数从大到小排序
    sorted_block_dict = dict(
        sorted(
            block_dict.items(),
            key = lambda item: (item[1]['data'].size(1), item[1]['data'].size(0)),
            reverse = True
        )
    )
    return sorted_block_dict


def get_cim_devices(model):
    # Initialize a list to store the layers to be replaced (name and new module)
    device_list = []

    # Iterate over all modules in the model to collect information about the layers to be replaced
    for name, module in model.named_modules():
        # Check if the current module is a convolutional or linear layer, and replace accordingly
        if type(module) in reg_dict.cim_layers:
            cim_device = module.layer_config['rram_device']
            device_list.append(cim_device)
    device_list = list(set(device_list))
    return device_list


def generate_weight_est(weight_tar_fp, std_tar):
    # 避免 log(0) 产生 NaN
    weight_tar_fp[weight_tar_fp == 0] += 0.01

    # 获取 weight_tar_fp 的唯一值索引，用于快速查找 std
    unique_values = torch.tensor(list(std_tar.keys()), dtype = torch.float32, device = weight_tar_fp.device)
    max_weight = torch.max(weight_tar_fp)
    min_weight = torch.min(weight_tar_fp)

    std_values = torch.tensor(list(std_tar.values()), dtype = torch.float32, device = weight_tar_fp.device)

    # 创建映射表：找到 weight_tar_fp 中每个元素在 unique_values 里的索引
    indices = torch.bucketize(weight_tar_fp.round(), unique_values, right = True) - 1

    # 确保索引合法，防止 out-of-bounds
    indices = torch.clamp(indices, 0, len(unique_values) - 1)

    # 获取每个元素对应的 std 值
    weight_std = std_values[indices]

    # 生成随机扰动项（高斯分布）
    noise = torch.randn_like(weight_tar_fp, device = weight_tar_fp.device) * weight_std

    # 计算新权重
    weight_est_log = torch.log(torch.abs(weight_tar_fp)) + noise
    weight_est = torch.exp(weight_est_log)

    # 恢复负数权重的符号
    weight_est[weight_tar_fp < 0] *= -1

    # 计算 std_est
    std_est = (weight_est - weight_tar_fp).std() / (max_weight - min_weight)

    return weight_est, std_est


def weight_drift(model,
                 std_tar,
                 module_for_map = reg_dict.custom_layers,
                 exclude_layers = None,
                 assign_layers = None):
    for name, module in model.named_modules():
        if type(module) in module_for_map:
            if exclude_layers is not None:
                # Skip layers listed in excluded_layers
                if name in exclude_layers:
                    continue
            elif assign_layers is not None:
                # Only include layers listed in assign_layers
                if name not in assign_layers:
                    continue
            module.weight.data, std_est = generate_weight_est(module.weight.data, std_tar)
            print(f'Generated Drifted Weight: {name}')
            print(f'Effective std = {std_est}')
def map_weight_for_model(model,
                         array_size,
                         weight_block_size,
                         array_device_name = 'array',
                         draw_weight_block = True,
                         assign_layers = None,
                         exclude_layers = None,
                         module_for_map = reg_dict.custom_layers):
    weight_info = convert_to_cim_weights(model = model,
                                         weight_block_size = weight_block_size,
                                         module_for_map = module_for_map,
                                         assign_layers = assign_layers,
                                         exclude_layers = exclude_layers,
                                         )
    block_dict, lut = weight_info_to_blocks(weight_info)
    block_dict = sort_block_dict(block_dict)
    block_mapping_info = map_blocks_to_boxes(block_dict = block_dict,
                                             box_size = array_size)
    model_weight_mapping_info = {}
    for block_key, lut_data in lut.items():
        for name, module in model.named_modules():
            # Check if the current module is a convolutional or linear layer, and replace accordingly
            if name == lut_data['layer_key']:
                split_key = lut_data['split_key']
                weight_addr = block_mapping_info[block_key]['weight_addr']
                array_idx = block_mapping_info[block_key]['array_idx']
                module.weight_mapping_info[split_key]['weight_addr'] = weight_addr
                module.weight_mapping_info[split_key]['array_idx'] = array_idx
                module.weight_mapping_info[split_key]['array_size'] = array_size
                module.array_device = array_device_name
                # ============================================================== #
                # 把 mapping 信息提取出来， 便于用户阅读，不会对变量产生影响
                # ============================================================== #
                model_weight_mapping_info[name] = module.weight_mapping_info
                module.name = name
                break
    if draw_weight_block:
        draw_weight_blocks(model, path = f'Array_Mapping_Info_({array_device_name})')
    return model_weight_mapping_info


def clear_directory(directory):
    """删除指定文件夹中的所有文件，避免目录不存在时报错"""
    if not os.path.exists(directory):
        # 如果目录不存在，直接返回，不报错
        return

    # 目录存在时，继续删除
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


# 画出权重mapping的位置信息
def draw_weight_blocks(model, path = None):
    if path is None:
        cur_dir = os.getcwd()
        dir = f'{cur_dir}/Arrays_Draw'
    else:
        dir = path

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        clear_directory(dir)  # 清空文件夹

    # 创建设备和对应数组的映射
    array_devices = {}

    for name, module in model.named_modules():
        if hasattr(module, 'weight_mapping_info'):
            array_device = module.array_device
            if array_device not in array_devices:
                array_devices[array_device] = []

    # 遍历所有的层并将信息存储到设备映射中
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mapping_info'):
            # 为层生成随机颜色
            h = random.random()
            # 使用colorsys将HSV值转换为RGB值
            r, g, b = colorsys.hsv_to_rgb(h, s = 0.8, v = 1)
            random_color = (r, g, b)

            # 生成低饱和度的颜色
            r_low, g_low, b_low = colorsys.hsv_to_rgb(h, s = 0.5, v = 1)
            low_saturation_color = (r_low, g_low, b_low)

            # 获取权重分割信息
            device_arrays = array_devices[module.array_device]
            for split_key, split_info in module.weight_mapping_info.items():
                if split_info.get('array_idx', None) != None:
                    array_idx = split_info['array_idx']
                    array_size = split_info['array_size']
                    # 存储每个设备和对应的层数组
                    while len(device_arrays) <= array_idx:
                        device_arrays.append(None)
                    if device_arrays[array_idx] is None:
                        device_arrays[array_idx] = {'size': array_size,
                                                    'layers': []}
                    # 添加层和颜色信息
                    device_arrays[array_idx]['layers'].append(
                        (name, split_info, random_color, low_saturation_color))

    # 遍历每个设备数组
    for array_device, device_arrays in array_devices.items():
        for array_idx, array_info in enumerate(device_arrays):
            if array_info is None:
                continue

            # 创建灰度图
            array = np.ones(array_info['size']) * 127
            plt.imshow(array, cmap = 'gray',
                       extent = (0, array_info['size'][1], array_info['size'][0], 0))  # 使用灰度图
            plt.axis('off')  # 关闭坐标轴

            # 遍历数组的层并绘制权重区域
            for (layer_name,
                 split_info,
                 random_color,
                 low_saturation_color) in array_info['layers']:
                start_row, start_col, _, _ = split_info['weight_addr']
                row_num, col_num = split_info['row_num'], split_info['col_num']

                # 在区域外侧画框线
                plt.plot(
                    [start_col, start_col + col_num, start_col + col_num, start_col, start_col],
                    [start_row, start_row, start_row + row_num, start_row + row_num, start_row],
                    color = random_color, linewidth = 1)

                # 使用低饱和度的颜色填充框内部
                plt.fill_between(
                    [start_col, start_col + col_num],
                    start_row, start_row + row_num,
                    color = low_saturation_color, alpha = 0.3
                )

                # 在区域内部标注层名
                plt.text(start_col + col_num / 2, start_row + row_num / 2, layer_name,
                         color = random_color, fontsize = 8,
                         horizontalalignment = 'center', verticalalignment = 'center')

            # 保存图像为jpg
            plt.tight_layout(h_pad = 0, w_pad = 0)
            plt.savefig(f"{dir}/{array_device}_{array_idx}.jpg", format = 'jpg',
                        dpi = 300, bbox_inches = 'tight', pad_inches = 0)
            plt.close()


def draw_weight_blocks_idx_name(model, path = None):
    if path is None:
        cur_dir = os.getcwd()
        dir = f'{cur_dir}/Arrays_Draw'
    else:
        dir = path

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        clear_directory(dir)  # 清空文件夹
    # 创建设备和对应数组的映射
    device_arrays = []

    name_idx = 0
    # 遍历所有的层并将信息存储到设备映射中
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mapping_info'):
            # 为层生成随机颜色
            h = random.random()
            # 使用colorsys将HSV值转换为RGB值
            r, g, b = colorsys.hsv_to_rgb(h, s = 0.8, v = 1)
            random_color = (r, g, b)
            # 获取权重分割信息

            for split_key, split_info in module.weight_mapping_info.items():
                if split_info.get('array_idx', None) != None:
                    array_idx = split_info['array_idx']
                    array_size = split_info['array_size']
                    # 存储每个设备和对应的层数组
                    while len(device_arrays) <= array_idx:
                        device_arrays.append(None)
                    if device_arrays[array_idx] is None:
                        device_arrays[array_idx] = {'size': array_size,
                                                    'layers': []}
                    # 添加层和颜色信息
                    device_arrays[array_idx]['layers'].append(
                        (f'Conv_{name_idx}', split_info, random_color))
            name_idx += 1
    # 遍历每个设备数组
    for array_idx, array_info in enumerate(device_arrays):
        if array_info is None:
            continue

        # 创建灰度图
        array = np.ones(array_info['size']) * 127
        plt.imshow(array, cmap = 'gray',
                   extent = (0, array_info['size'][1], array_info['size'][0], 0))  # 使用灰度图
        plt.axis('off')  # 关闭坐标轴

        # 遍历数组的层并绘制权重区域
        for layer_name, split_info, random_color in array_info['layers']:
            start_row, start_col, _, _ = split_info['weight_addr']
            row_num, col_num = split_info['row_num'], split_info['col_num']

            # 在区域外侧画框线
            plt.plot(
                [start_col, start_col + col_num, start_col + col_num, start_col, start_col],
                [start_row, start_row, start_row + row_num, start_row + row_num, start_row],
                color = random_color, linewidth = 1)

            # 在区域内部标注层名
            plt.text(start_col + col_num / 2, start_row + row_num / 2, layer_name,
                     color = random_color, fontsize = 8,
                     horizontalalignment = 'center', verticalalignment = 'center')

        # 保存图像为jpg
        plt.tight_layout(h_pad = 0, w_pad = 0)
        plt.savefig(f"{dir}/array_{array_idx}.jpg", format = 'jpg',
                    dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.close()
