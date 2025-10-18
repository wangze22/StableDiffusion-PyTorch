import colorsys
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


# block_dict = {
#     block_key: {
#         'data': weight_data
#     },
#     block_key: {
#         'data': weight_data
#     },
#     block_key: {
#         'data': weight_data
#     },
# }

def map_blocks_to_boxes(block_dict, box_size):
    # cur_dir = os.getcwd()
    def is_larger(A, B):
        # 获取A和B的行数和列数
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape

        # 判断A是否在行或列的尺寸上大于B
        return rows_A > rows_B or cols_A > cols_B

    def convert_matrix_to_tuples(matrix):
        return [(val, i) for i, val in enumerate(matrix)]

    # 根据 search_points，找到每行的所有连续空间（可继续放置权重的空间）中最小的col，
    # 作为下个权重放置时的其中一个搜索的点位
    def find_left_most_col_for_row(search_points, target_row, start_col):
        while start_col > 0:
            # 查找在当前列的左侧是否有更低的高度
            for point in search_points:
                if point[0] < target_row and point[1] == start_col - 1:
                    start_col -= 1
                    break
            else:
                # 如果没有找到更低的高度，那么停止搜索
                break
        return start_col

    def add_array(array_info):
        idx = len(array_info)
        array_info[idx] = {}
        array_info[idx]['array_data'] = torch.zeros(box_size)
        array_info[idx]['array_usage'] = np.zeros(box_size, dtype = bool)
        array_info[idx]['height_map'] = np.zeros(box_size[1], dtype = int)
        array_info[idx]['search_points'] = [(0, 0)]
        array_info[idx]['has_data'] = False

    # 根据高度图,找到每个高度下可以继续放置权重的点
    def find_first_col(height_map):
        search_points = convert_matrix_to_tuples(height_map)
        search_points.sort(key = lambda x: (x[0], x[1]))

        new_values = []
        continuous_start_col = None
        previous_row = None
        previous_col = None

        for point in search_points:
            row, col = point
            # 如果这是开始或者行改变
            if previous_row is None or row != previous_row:
                if continuous_start_col is not None:
                    # 使用更低的高度进行调整
                    adjusted_start_col = find_left_most_col_for_row(search_points, previous_row,
                                                                    continuous_start_col)
                    new_values.append((previous_row, adjusted_start_col))
                continuous_start_col = col
            # 如果列不是连续的或行改变
            elif col != previous_col + 1:
                # 使用更低的高度进行调整
                adjusted_start_col = find_left_most_col_for_row(search_points, previous_row,
                                                                continuous_start_col)
                new_values.append((previous_row, adjusted_start_col))
                continuous_start_col = col
            previous_row, previous_col = point

        # 处理最后一个连续区域
        if continuous_start_col is not None:
            adjusted_start_col = find_left_most_col_for_row(search_points, previous_row, continuous_start_col)
            new_values.append((previous_row, adjusted_start_col))

        return new_values

    if len(block_dict) == 0:
        print(f'No Weight For Mapping')
        return
    array_data = {}
    array_usage = {}
    search_points = {}
    height_map = {}
    # TODO
    # 此处加入一个函数，把all_weights中的所有rram_device的类型提取出来，例如144k 576k等等。
    # 然后再加一个循环,先对144k的所有权重提取出来,按照从大到小排序,对其进行mapping,然后再把576k的提取出来mapping
    for key, block in block_dict.items():
        block['placed'] = False

    # 初始化以下变量
    # device_arrays
    # device_usage
    # array_idx_counter
    # height_map
    # search_points

    array_info = {}

    add_array(array_info)

    array_idx = 0

    block_mapping_info = {}
    # 开始部署权重
    while not all(block['placed'] for block_key, block in block_dict.items()):
        # 如果 full_search 在循环结束后仍为Ture，说明对于所有point和weight，该array都没有位置了
        # 因此直接创建一个新的array
        full_search = True
        search_points = array_info[array_idx]['search_points']
        array_data = array_info[array_idx]['array_data']
        array_usage = array_info[array_idx]['array_usage']
        height_map = array_info[array_idx]['height_map']

        for point in search_points:
            (row, col) = point
            for block_key, block in block_dict.items():
                placed_info = block['placed']
                if placed_info:
                    continue
                placed = placed_info
                block_data = block['data']
                # weight_half_level = model_info[layer_key]['layer_config']['weight_half_level']
                # assert abs(weight_data).max() <= weight_half_level
                if is_larger(block_data, array_data):
                    print(f'Weight is Larger than Array. Unable to map.')
                    exit(1)
                # 检查行是否超出范围
                is_row_in_range = row + block_data.shape[0] <= array_data.shape[0]
                # 检查列是否超出范围
                is_col_in_range = col + block_data.shape[1] <= array_data.shape[1]
                # 定义要检查的使用范围
                start_row, end_row = row, row + block_data.shape[0]
                start_col, end_col = col, col + block_data.shape[1]
                # 检查指定的范围是否已被使用
                is_area_unused = not np.any(array_usage[start_row:end_row, start_col:end_col])

                if is_row_in_range and is_col_in_range and is_area_unused:
                    # for weight_same_layer in all_weights:
                    #     if weight_same_layer['layer_key'] == layer_key:
                    # 当找到一个位置后，插入权重并更新 height_map 和 search_points
                    # 更新device_arrays的指定范围
                    target_array_section = array_data[start_row:end_row, start_col:end_col]
                    target_array_section[:] = block_data

                    # 更新device_usage的指定范围
                    usage_section = array_usage[start_row:end_row, start_col:end_col]
                    usage_section[:] = True

                    start_col = col
                    end_col = col + block_data.shape[1]

                    height_map[start_col:end_col] = np.maximum(
                        height_map[start_col:end_col],
                        row + block_data.shape[0])

                    block_mapping_info[block_key] = {}
                    block_mapping_info[block_key]['weight_addr'] = (
                        int(row), int(col), int(block_data.shape[0]), int(block_data.shape[1]))

                    block_mapping_info[block_key]['array_idx'] = array_idx

                    placed = True
                    block['placed'] = placed

                    # 更新 search_points
                    search_points = find_first_col(height_map)
                    array_info[array_idx]['search_points'] = search_points
                    array_info[array_idx]['has_data'] = True
                    # search_points[rram_device] = sorted(search_points[rram_device], key = lambda x: (x[0], x[1]))
                    break

            if placed:
                full_search = False
                break

        # 两个判断条件：
        # 1. 如果 full_search 在循环结束后仍为Ture，说明对于所有point和weight，该array都没有位置了
        # 2. 所有权重都已经被放置过了
        # 满足任意一个，则保存当前array并新建一个
        if full_search or all(block['placed'] for _, block in block_dict.items()):
            # 保存权重npy文件
            # cur_dir = os.getcwd()
            # dir = f'{cur_dir}/Arrays'
            # if not os.path.exists(dir):
            #     os.makedirs(dir)
            # array_file_name = f"{dir}/{rram_device}_array_{array_idx}.npy"
            # np.save(array_file_name, array_data)
            add_array(array_info)
            array_idx += 1

    return block_mapping_info


