# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

import time

from cim_qn_train.layers_LSQ import *
from cim_qn_train.layers_LSQ_bit_split import *
if __name__ == '__main__':
    torch.manual_seed(0)
    # ====================================== #
    # 卷积层测试
    # ====================================== #
    tt = Conv2d_quant_noise_LSQ_bit_split(
        in_channels = 100,
        out_channels = 30,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        groups = 1,
        weight_bit = 4,
        input_bit = 8,
        output_bit = 8,
        dac_bit = 3,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    tt2 = Conv2d_quant_noise_LSQ(
        in_channels = 100,
        out_channels = 30,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        groups = 1,
        weight_bit = 4,
        input_bit = 8,
        output_bit = 8,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )
    tt.to(torch.device('cuda'))
    tt2.to(torch.device('cuda'))

    x = torch.rand((30, 100, 50, 50), device = torch.device('cuda'))
    y1 = tt(x)

    t_adda = time.time()
    for i in range(100):
        y1 = tt(x)
    t_adda = time.time() - t_adda
    print(f't_adda = {t_adda:.2f}s')

    tt_state_dict = tt.state_dict()
    tt2.load_state_dict(tt_state_dict)

    t_lsq = time.time()
    for i in range(100):
        y2 = tt2(x)
    t_lsq = time.time() - t_lsq
    print(f't_lsq = {t_lsq:.2f}s')

    diff = y1 - y2
    max_diff = torch.max(diff)
    print(f'max_diff = {max_diff}')
    non_zero_count = torch.sum(max_diff != 0).item()
    print(non_zero_count)

    time_diff = t_adda /t_lsq
    print(f'time_diff = {time_diff:.2f}')

    # ====================================== #
    # 全连接层测试
    # ====================================== #
    tt = Linear_quant_noise_LSQ_bit_split(
        in_features = 1000,
        out_features = 1000,
        weight_bit = 4,
        input_bit = 16,
        output_bit = 8,
        dac_bit = 3,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    tt2 = Linear_quant_noise_LSQ(
        in_features = 1000,
        out_features = 1000,
        weight_bit = 4,
        input_bit = 16,
        output_bit = 8,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )
    tt.to(torch.device('cuda'))
    tt2.to(torch.device('cuda'))

    x = torch.rand((3000, 1000)).to(torch.device('cuda'))
    y1 = tt(x)

    t_adda = time.time()
    for i in range(100):
        y1 = tt(x)
    t_adda = time.time() - t_adda
    print(f't_adda = {t_adda:.2f}s')

    tt_state_dict = tt.state_dict()
    tt2.load_state_dict(tt_state_dict)

    t_lsq = time.time()
    for i in range(100):
        y2 = tt2(x)
    t_lsq = time.time() - t_lsq
    print(f't_lsq = {t_lsq:.2f}s')

    diff = y1 - y2
    max_diff = torch.max(diff)
    print(f'max_diff = {max_diff}')
    non_zero_count = torch.sum(max_diff != 0).item()
    print(non_zero_count)

    time_diff = t_adda / t_lsq
    print(f'time_diff = {time_diff:.2f}')
