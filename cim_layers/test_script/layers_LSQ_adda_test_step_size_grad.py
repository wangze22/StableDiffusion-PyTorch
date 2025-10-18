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
    adda_net = Conv2d_lsq_adda(
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
        adc_bit = 8,
        adc_k = 1/50,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    lsq_net = Conv2d_qn_lsq(
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
    adda_net.to(torch.device('cuda'))
    lsq_net.to(torch.device('cuda'))

    x = torch.rand((30, 100, 50, 50),
                   device = torch.device('cuda'))
    adda_out = adda_net(x)

    tt_state_dict = adda_net.state_dict()
    lsq_net.load_state_dict(tt_state_dict, strict = False)

    lsq_out = lsq_net(x)

    diff_y = adda_out - lsq_out

    fm_non_zero_count = torch.sum(diff_y != 0).item()
    print(f'fm_non_zero_count = {fm_non_zero_count}')
    print(f'percent = {fm_non_zero_count / diff_y.numel() * 100:.2f}%')

    fm_relative_diff = abs(diff_y) / abs(lsq_out).max()
    # print(f'fm_relative_diff = {fm_relative_diff}')
    fm_relative_diff_max = fm_relative_diff.max()
    print(f'fm_relative_diff_max = {fm_relative_diff_max}')


    target = torch.randn_like(adda_out)
    loss_1 = (target - adda_out).pow(2).mean()
    loss_2 = (target - lsq_out).pow(2).mean()
    loss_1.backward()
    loss_2.backward()

    print(f'==================================================')
    weight_grad_adda = adda_net.weight.grad
    weight_grad_lsq = lsq_net.weight.grad
    grad_diff = weight_grad_adda - weight_grad_lsq
    grad_max_diff = torch.max(grad_diff.abs())
    print(weight_grad_adda[0][0])
    print(weight_grad_lsq[0][0])
    print(f'weight_grad_max_diff_relative = {grad_max_diff / torch.max(weight_grad_lsq.abs()) *100:.2f}%')
    non_zero_count = torch.sum(grad_diff != 0).item()
    print(f'weight_grad non_zero_count = {non_zero_count}')
    print(f'==================================================')

    w_grad_adda = adda_net.step_size_weight.grad
    w_grad_lsq = lsq_net.step_size_weight.grad
    print(f'w_grad_adda = {w_grad_adda}')
    print(f'w_grad_lsq = {w_grad_lsq}')
    print(f'ratio = {w_grad_adda/w_grad_lsq}')
    print(f'==================================================')
    in_grad_adda = adda_net.step_size_input.grad
    in_grad_lsq = lsq_net.step_size_input.grad
    print(f'in_grad_adda = {in_grad_adda}')
    print(f'in_grad_lsq = {in_grad_lsq}')
    print(f'ratio = {in_grad_adda/in_grad_lsq}')
    print(f'==================================================')
    out_grad_adda = adda_net.step_size_output.grad
    out_grad_lsq = lsq_net.step_size_output.grad
    print(f'out_grad_adda = {out_grad_adda}')
    print(f'out_grad_lsq = {out_grad_lsq}')
    print(f'ratio = {out_grad_adda/out_grad_lsq}')
    print(f'==================================================')
    adc_gain_grad_adda = adda_net.adc_gain.grad
    print(f'adc_gain_grad_adda = {adc_gain_grad_adda}')
    # ====================================== #
    # 全连接层测试
    # ====================================== #
    # print('=================================================')
    # tt = Linear_lsq_adda(
    #     in_features = 100,
    #     out_features = 1000,
    #     weight_bit = 4,
    #     input_bit = 16,
    #     output_bit = 8,
    #     dac_bit = 2,
    #     adc_bit = 40,
    #     adc_k = 1,
    #     noise_scale = 0,
    #     gain_noise_scale = 0,
    #     offset_noise_scale = 0,
    #     bias = True
    # )
    #
    # tt2 = Linear_qn_lsq(
    #     in_features = 100,
    #     out_features = 1000,
    #     weight_bit = 4,
    #     input_bit = 16,
    #     output_bit = 8,
    #     noise_scale = 0,
    #     gain_noise_scale = 0,
    #     offset_noise_scale = 0,
    #     bias = True
    # )
    # tt.to(torch.device('cuda'))
    # tt2.to(torch.device('cuda'))
    #
    # x = torch.rand((3000, 100)).to(torch.device('cuda'))
    # y1 = tt(x)
    #
    # tt_state_dict = tt.state_dict()
    # tt2.load_state_dict(tt_state_dict, strict = False)
    #
    # y2 = tt2(x)
    #
    # diff_y = y1 - y2
    #
    # fm_non_zero_count = torch.sum(diff_y != 0).item()
    # print(f'fm_non_zero_count = {fm_non_zero_count}')
    # print(f'percent = {fm_non_zero_count / diff_y.numel() * 100:.2f}%')
    #
    # fm_relative_diff = abs(diff_y) / abs(y2).max()
    # # print(f'fm_relative_diff = {fm_relative_diff}')
    # fm_relative_diff_max = fm_relative_diff.max()
    # print(f'fm_relative_diff_max = {fm_relative_diff_max}')
    #
    # target = torch.randn_like(y1)
    # loss_1 = (target - y1).pow(2).mean()
    # loss_2 = (target - y2).pow(2).mean()
    # loss_1.backward()
    # loss_2.backward()
    #
    # grad_1 = tt.weight.grad
    # grad_2 = tt2.weight.grad
    # grad_diff = grad_1 - grad_2
    # grad_max_diff = torch.max(grad_diff)
    # print(f'grad_max_diff = {grad_max_diff}')
    # non_zero_count = torch.sum(grad_diff != 0).item()
    # print(f'grad non_zero_count = {non_zero_count}')




