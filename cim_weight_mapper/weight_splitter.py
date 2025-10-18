import numpy as np
import math

import torch
import math


def split_weight(weight_2d, block_size):

    max_rows = block_size[0]
    max_cols = block_size[1]

    rows, cols = weight_2d.shape

    # 权重切分数量
    num_blocks_row = rows // max_rows
    num_blocks_col = cols // max_cols

    weight_split_info = {}

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
            weight_data = weight_2d[start_row:start_row + actual_rows, start_col:start_col + actual_cols]

            weight_split_info[key] = {
                "start_row": start_row,
                "start_col": start_col,
                "row_num": actual_rows,
                "col_num": actual_cols,
                "weight_data": weight_data,
            }

    return weight_split_info