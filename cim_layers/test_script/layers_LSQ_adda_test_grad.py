# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

from cim_layers.layers_qn_lsq import *
from cim_layers.layers_qn_lsq_adda import *

if __name__ == '__main__':
    torch.manual_seed(0)
    # ====================================== #
    # 卷积层测试
    # ====================================== #
    tt = Conv2d_lsq_adda(
        in_channels = 100,
        out_channels = 30,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        groups = 1,
        weight_bit = 4,
        input_bit = 8,
        output_bit = 8,
        dac_bit = 2,
        adc_bit = 40,
        adc_k = 1,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    tt2 = Conv2d_qn_lsq(
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

    tt_state_dict = tt.state_dict()
    tt2.load_state_dict(tt_state_dict, strict = False)

    y2 = tt2(x)

    diff_y = y1 - y2

    fm_non_zero_count = torch.sum(diff_y != 0).item()
    print(f'fm_non_zero_count = {fm_non_zero_count}')
    print(f'percent = {fm_non_zero_count / diff_y.numel() * 100:.2f}%')

    fm_relative_diff = abs(diff_y) / abs(y2).max()
    # print(f'fm_relative_diff = {fm_relative_diff}')
    fm_relative_diff_max = fm_relative_diff.max()
    print(f'fm_relative_diff_max = {fm_relative_diff_max}')


    target = torch.randn_like(y1)
    loss_1 = (target - y1).pow(2).mean()
    loss_2 = (target - y2).pow(2).mean()
    loss_1.backward()
    loss_2.backward()

    grad_1 = tt.weight.grad
    grad_2 = tt2.weight.grad
    grad_diff = grad_1 - grad_2
    grad_max_diff = torch.max(grad_diff)
    print(f'grad_max_diff = {grad_max_diff}')
    non_zero_count = torch.sum(grad_diff != 0).item()
    print(f'grad non_zero_count = {non_zero_count}')

    # ====================================== #
    # 全连接层测试
    # ====================================== #
    print('=================================================')
    tt = Linear_lsq_adda(
        in_features = 100,
        out_features = 1000,
        weight_bit = 4,
        input_bit = 16,
        output_bit = 8,
        dac_bit = 2,
        adc_bit = 40,
        adc_k = 1,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    tt2 = Linear_qn_lsq(
        in_features = 100,
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

    x = torch.rand((3000, 100)).to(torch.device('cuda'))
    y1 = tt(x)

    tt_state_dict = tt.state_dict()
    tt2.load_state_dict(tt_state_dict, strict = False)

    y2 = tt2(x)

    diff_y = y1 - y2

    fm_non_zero_count = torch.sum(diff_y != 0).item()
    print(f'fm_non_zero_count = {fm_non_zero_count}')
    print(f'percent = {fm_non_zero_count / diff_y.numel() * 100:.2f}%')

    fm_relative_diff = abs(diff_y) / abs(y2).max()
    # print(f'fm_relative_diff = {fm_relative_diff}')
    fm_relative_diff_max = fm_relative_diff.max()
    print(f'fm_relative_diff_max = {fm_relative_diff_max}')

    target = torch.randn_like(y1)
    loss_1 = (target - y1).pow(2).mean()
    loss_2 = (target - y2).pow(2).mean()
    loss_1.backward()
    loss_2.backward()

    grad_1 = tt.weight.grad
    grad_2 = tt2.weight.grad
    grad_diff = grad_1 - grad_2
    grad_max_diff = torch.max(grad_diff)
    print(f'grad_max_diff = {grad_max_diff}')
    non_zero_count = torch.sum(grad_diff != 0).item()
    print(f'grad non_zero_count = {non_zero_count}')




