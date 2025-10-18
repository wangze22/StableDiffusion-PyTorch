# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""

import time

from cim_layers.layers_qn_lsq import *
from cim_layers.layers_qn_lsq_bit_split import *
if __name__ == '__main__':
    torch.manual_seed(0)
    batch = 10
    in_features = 10
    out_features = 10
    # ====================================== #
    # 卷积层测试
    # ====================================== #
    bs_net = Linear_quant_noise_LSQ_bit_split(
        in_features = in_features,
        out_features = out_features,
        weight_bit = 4,
        input_bit = 8,
        output_bit = 8,
        dac_bit = 2,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = False
    )

    lsq_net = Linear_qn_lsq(
        in_features = in_features,
        out_features = out_features,
        weight_bit = 4,
        input_bit = 8,
        output_bit = 8,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = False
    )
    # lsq_net.weight.data = torch.ones_like(lsq_net.weight.data)

    bs_net.to(torch.device('cuda'))
    lsq_net.to(torch.device('cuda'))

    x = torch.rand((batch,in_features),
                   device = torch.device('cuda'))
    adda_out = bs_net(x)

    tt_state_dict = bs_net.state_dict()
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
    weight_grad_adda = bs_net.weight.grad
    weight_grad_lsq = lsq_net.weight.grad
    grad_diff = weight_grad_adda - weight_grad_lsq
    grad_max_diff = torch.max(grad_diff.abs())
    print(weight_grad_adda)
    print(weight_grad_lsq)
    print(f'weight_grad_max_diff_relative = {grad_max_diff / torch.max(weight_grad_lsq.abs()) *100:.2f}%')
    non_zero_count = torch.sum(grad_diff != 0).item()
    print(f'weight_grad non_zero_count = {non_zero_count}')
    print(f'==================================================')

    w_grad_adda = bs_net.step_size_weight.grad
    w_grad_lsq = lsq_net.step_size_weight.grad
    print(f'w_grad_adda = {w_grad_adda}')
    print(f'w_grad_lsq = {w_grad_lsq}')
    print(f'ratio = {w_grad_adda/w_grad_lsq}')
    print(f'==================================================')
    in_grad_adda = bs_net.step_size_input.grad
    in_grad_lsq = lsq_net.step_size_input.grad
    print(f'in_grad_adda = {in_grad_adda}')
    print(f'in_grad_lsq = {in_grad_lsq}')
    print(f'ratio = {in_grad_adda/in_grad_lsq}')
    print(f'==================================================')
    out_grad_adda = bs_net.step_size_output.grad
    out_grad_lsq = lsq_net.step_size_output.grad
    print(f'out_grad_adda = {out_grad_adda}')
    print(f'out_grad_lsq = {out_grad_lsq}')
    print(f'ratio = {out_grad_adda/out_grad_lsq}')
    print(f'==================================================')
    # # ====================================== #
    # # 全连接层测试
    # # ====================================== #
    # tt = Linear_quant_noise_LSQ_bit_split(
    #     in_features = 1000,
    #     out_features = 1000,
    #     weight_bit = 4,
    #     input_bit = 16,
    #     output_bit = 8,
    #     dac_bit = 3,
    #     noise_scale = 0,
    #     gain_noise_scale = 0,
    #     offset_noise_scale = 0,
    #     bias = True
    # )
    #
    # tt2 = Linear_quant_noise_LSQ(
    #     in_features = 1000,
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
    # x = torch.rand((3000, 1000)).to(torch.device('cuda'))
    # y1 = tt(x)
    #
    # tt_state_dict = tt.state_dict()
    # tt2.load_state_dict(tt_state_dict)
    #
    # y2 = tt2(x)
    #
    # target = torch.randn_like(y1)
    # loss_1 = (target - y1).pow(2).mean()
    # loss_2 = (target - y2).pow(2).mean()
    # loss_1.backward()
    # loss_2.backward()
    #
    # grad_1 = tt.weight.grad
    # grad_2 = tt2.weight.grad
    # diff = grad_1 - grad_2
    #
    # print(grad_1)
    # print(grad_2)
    #
    # max_diff = torch.max(diff)
    # print(f'max_diff = {max_diff}')
    # non_zero_count = torch.sum(max_diff != 0).item()
    # print(non_zero_count)