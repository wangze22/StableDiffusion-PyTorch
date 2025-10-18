# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
from memory_profiler import profile
import cProfile
import pstats
from cim_layers.layers_qn_lsq import *
from cim_layers.layers_qn_lsq_adda_cim import *
from cim_weight_mapper.weight_process import *
torch.manual_seed(0)
# ====================================== #
# 卷积层测试
# ====================================== #
in_channel = 32
out_channel = 32
img_size = 64
batch = 32
array_size = [49,128]
device = 'cpu'
tar = Conv2d_qn_lsq(
    in_channels = in_channel,
    out_channels = out_channel,
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
tar.to(torch.device(device))

tt = Conv2d_lsq_adda_cim(
    in_channels = in_channel,
    out_channels = out_channel,
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
tt.to(torch.device(device))

x = torch.rand((batch, in_channel, img_size, img_size), device = torch.device(device))
tar_out = tar(x)

tar_state_dict = tar.state_dict()
tt.load_state_dict(tar_state_dict, strict = False)

map_weight_for_model(tt, array_size = array_size, weight_block_size = array_size)
draw_weight_blocks(tt)
@profile
def test_func():
    y = tt(x)

if __name__ == '__main__':
    test_func()
    # profile = cProfile.Profile()
    # profile.run('test_func()')
    # profile.dump_stats('profile_output.prof')
    #
    # # 查看分析结果
    # with open('profile_output.txt', 'w') as f:
    #     ps = pstats.Stats('profile_output.prof', stream = f)
    #     ps.strip_dirs().sort_stats('tottime').print_stats()

    # y = tt(x)
    # diff_y = y - tar_out
    #
    # fm_non_zero_count = torch.sum(diff_y != 0).item()
    # print(f'fm_non_zero_count = {fm_non_zero_count}')
    # print(f'percent = {fm_non_zero_count / diff_y.numel() * 100:.2f}%')
    #
    # fm_relative_diff = abs(diff_y) / abs(tar_out).max()
    # # print(f'fm_relative_diff = {fm_relative_diff}')
    # fm_relative_diff_max = fm_relative_diff.max()
    # print(f'fm_relative_diff_max = {fm_relative_diff_max}')
    #
    # target = torch.randn_like(tar_out)
    # loss_1 = (target - tar_out).pow(2).mean()
    # loss_2 = (target - y).pow(2).mean()
    # loss_1.backward()
    # loss_2.backward()
    #
    # grad_1 = tt.weight.grad
    # grad_2 = tar.weight.grad
    # print(f'grad_1 = {grad_1[0][0]}')
    # print(f'grad_2 = {grad_2[0][0]}')
    # grad_diff = grad_1 - grad_2
    # grad_max_diff = torch.max(grad_diff)
    # print(f'grad_max_diff = {grad_max_diff}')
    # grad_max_diff_relative = grad_max_diff / (tar.weight.grad).max()
    # print(f'grad_max_diff_relative = {grad_max_diff_relative}')
    # non_zero_count = torch.sum(grad_diff != 0).item()
    # print(f'grad non_zero_count = {non_zero_count}')
    # print(f'grad non_zero_count = {non_zero_count / tt.weight.data.numel() * 100:.2f}%')

    # # ====================================== #
    # # 全连接层测试
    # # ====================================== #
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
    #
    #
    #
    #
