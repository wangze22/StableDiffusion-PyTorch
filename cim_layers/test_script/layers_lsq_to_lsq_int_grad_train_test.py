# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:26:50 2021

@author: Wang Ze
"""
import torch

from cim_layers.layers_lsq_int import *
from cim_layers.layers_qn_lsq import *
from cim_toolchain_utils.utils import *
from cim_layers.DDFP_optimizer import SGDWithMomentum

if __name__ == '__main__':
    # torch.manual_seed(0)
    batch_size = 20
    in_channels = 32
    out_channels = 32
    in_features = 100
    out_features = 100

    weight_bit = 8
    input_bit = 8
    output_bit = 8
    extended_bit = 4
    # ====================================== #
    # 训练参数
    # ====================================== #
    epochs = 1000
    lr = 1e-3
    patience = 200
    factor = 0.5
    min_lr = 1e-8
    # ====================================== #
    # 卷积层测试
    # ====================================== #
    print('=================================================')
    print('卷积层测试')
    print('=================================================')
    # ======================= #
    # LSQ 计算层
    # ======================= #
    lsq_net = Conv2d_qn_lsq(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        groups = 1,
        weight_bit = weight_bit,
        input_bit = input_bit,
        output_bit = output_bit,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    lsq_net2 = Conv2d_qn_lsq(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        groups = 1,
        weight_bit = weight_bit,
        input_bit = input_bit,
        output_bit = output_bit,
        noise_scale = 0,
        gain_noise_scale = 0,
        offset_noise_scale = 0,
        bias = True
    )

    lsq_net.to(torch.device('cuda'))
    lsq_net2.to(torch.device('cuda'))
    x = torch.rand((batch_size, in_channels, 50, 50),
                   device = torch.device('cuda'))

    lsq_out = lsq_net(x)
    lsq_out2 = lsq_net2(x)
    # ======================= #
    # LSQ INT 计算层
    # ======================= #
    lsq_int_net = Conv2d_lsq_int(copy.deepcopy(lsq_net2),
                                 int_grad = True,
                                 weight_bit_extension = extended_bit)
    lsq_int_net.to(torch.device('cuda'))

    # for name, module in lsq_int_net.named_modules():
    #     if type(module) == Identity_layer:
    #         print(name)

    x2 = x.clone().detach().requires_grad_(True)
    lsq_int_out = lsq_int_net(x2)

    diff_y = lsq_out - lsq_int_out

    fm_non_zero_count = torch.sum(diff_y != 0).item()
    print(f'fm_non_zero_count = {fm_non_zero_count}')
    print(f'percent = {fm_non_zero_count / diff_y.numel() * 100:.2f}%')

    fm_relative_diff = abs(diff_y) / abs(lsq_out).max()
    # print(f'fm_relative_diff = {fm_relative_diff}')
    fm_relative_diff_max = fm_relative_diff.max()
    print(f'fm_relative_diff_max = {fm_relative_diff_max}')

    scatter_plt(lsq_int_out, lsq_out)

    # ======================= #
    # 训练 LSQ INT 计算层
    # ======================= #
    lsq_net2.step_size_input.requires_grad_(False)
    lsq_int_net.lsq_quant_input.step_size.requires_grad_(False)

    lsq_net2.step_size_output.requires_grad_(False)
    lsq_int_net.lsq_quant_output.step_size.requires_grad_(False)

    lsq_net2.step_size_weight.requires_grad_(False)


    optimizer_int = torch.optim.Adam(lsq_int_net.parameters(), lr = lr)
    optimizer_lsq = torch.optim.Adam(lsq_net2.parameters(), lr = lr)

    # ======================= #
    # 使用自定义优化器
    # ======================= #
    step_size_w = lsq_int_net.lsq_quant_weight.step_size
    step_size_b = lsq_int_net.lsq_quant_bias.step_size
    step_size_list = [step_size_w, step_size_b]

    optimizer_int = SGDWithMomentum(lsq_int_net.parameters(), lsq_int_net.named_parameters(), lr = lr)


    scheduler_int = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_int, mode = 'min', factor = factor,
                                                               patience = patience, verbose = True, min_lr = min_lr)
    scheduler_lsq = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lsq, mode = 'min', factor = factor,
                                                               patience = patience, verbose = True, min_lr = min_lr)


    loss_history_int = []
    loss_history_lsq = []
    lsq_net.eval()
    for epoch in range(epochs):
        x = torch.rand((batch_size, in_channels, 50, 50),
                       device = torch.device('cuda'))


        lsq_out = lsq_net(x)

        lsq_int_out = lsq_int_net(x)
        lsq_out2 = lsq_net2(x)

        loss_int = torch.sum((lsq_out.detach() - lsq_int_out) ** 2)
        loss_lsq = torch.sum((lsq_out.detach() - lsq_out2) ** 2)

        optimizer_int.zero_grad()
        loss_int.backward()


        optimizer_lsq.zero_grad()
        loss_lsq.backward()

        # print(f'================================================================')
        # print(f'weight grad')
        # print(lsq_net2.weight.grad[0][0])
        # print(lsq_int_net.weight.grad[0][0])
        # scale = lsq_net2.weight.grad / lsq_int_net.weight.grad
        # print(f'scale grad = {scale}')
        # print(f'================================================================')
        # print(f'step_size_input grad')
        # print(lsq_net2.step_size_input.grad)
        # print(lsq_int_net.lsq_quant_input.step_size.grad)
        # print(f'================================================================')
        # print(f'step_size_output grad')
        # print(lsq_net2.step_size_output.grad)
        # print(lsq_int_net.lsq_quant_output.step_size.grad)
        # print(lsq_net2.step_size_output.grad / lsq_int_net.lsq_quant_output.step_size.grad)

        optimizer_int.step()
        # scheduler_int.step(loss_int)
        optimizer_lsq.step()
        # scheduler_lsq.step(loss_int)
        # w_grad = lsq_int_net.weight.grad
        # num_w_grad = unique_values(w_grad)
        # print(f'unique_values weight = {num_w_grad}')
        # print(lsq_int_net.weight.grad[0][0])
        # exit(0)

        # 使用 '\r' 实现同一行内更新
        print(f"\rEpoch {epoch + 1}/{epochs}, Loss int: {loss_int.item():.4f}, Loss lsq: {loss_lsq.item():.4f}", end = '')

        loss_history_int.append(loss_int.item())
        loss_history_lsq.append(loss_lsq.item())

    # 换行以确保最后一行输出完成后不会覆盖
    print()

    # 绘制 loss_history_int 和 loss_history_lsq
    plt.plot(range(1, epochs + 1), loss_history_int, marker = 'o', label = 'Loss int')
    plt.plot(range(1, epochs + 1), loss_history_lsq, marker = 'x', label = 'Loss lsq')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    scatter_plt(lsq_int_out, lsq_out, title = f'LSQ_INT')
    scatter_plt(lsq_out2, lsq_out, title = f'LSQ')
    # target = torch.randn_like(lsq_out)
    # loss_1 = (target - lsq_out).pow(2).mean()
    # loss_2 = (target - lsq_int_out).pow(2).mean()
    # loss_1.backward()
    # loss_2.backward()
    #
    # print(f'==================================================')
    # weight_grad_adda = qn_net.weight.grad
    # weight_grad_lsq = nn_net.weight.grad
    # grad_diff = weight_grad_adda - weight_grad_lsq
    # grad_max_diff = torch.max(grad_diff.abs())
    # print(weight_grad_adda[0][0])
    # print(weight_grad_lsq[0][0])
    # print(f'weight_grad_max_diff_relative = {grad_max_diff / torch.max(weight_grad_lsq.abs()) * 100:.2f}%')
    # non_zero_count = torch.sum(grad_diff != 0).item()
    # print(f'weight_grad non_zero_count = {non_zero_count}')
    # print(f'==================================================')
    # x_grad_test = x.grad
    # x_grad_tar = x2.grad
    # close = torch.isclose(x_grad_test, x_grad_tar, atol = 1e-5)
    # num_not_close = close.numel() - close.sum().item()
    # proportion_not_close = num_not_close / close.numel()
    # print(f'x_grad_test = {x_grad_test[0][0][0]}')
    # print(f'x_grad_tar = {x_grad_tar[0][0][0]}')
    # print(f'ratio = {x_grad_test[0][0][0] / x_grad_tar[0][0][0]}')
    # print(f'proportion_not_close = {proportion_not_close}')
    # print(f'==================================================')
    # # ====================================== #
    # # 全连接层测试
    # # ====================================== #
    # print('=================================================')
    # print('全连接层测试')
    # print('=================================================')
    # test_fc = Linear_lsq_DDFP_bp(
    #     in_features = in_features,
    #     out_features = out_features,
    #     weight_bit = weight_bit,
    #     input_bit = input_bit,
    #     output_bit = output_bit,
    #     noise_scale = 0,
    #     gain_noise_scale = 0,
    #     offset_noise_scale = 0,
    #     bias = True
    # )
    #
    # tar_fc = nn.Linear(
    #     in_features = in_features,
    #     out_features = out_features,
    #     bias = True
    # )
    # test_fc.to(torch.device('cuda'))
    # tar_fc.to(torch.device('cuda'))
    #
    # x = torch.rand((batch_size, 100),
    #                device = torch.device('cuda'), requires_grad = True)
    # test_out = test_fc(x)
    #
    # tt_state_dict = test_fc.state_dict()
    # tar_fc.load_state_dict(tt_state_dict, strict = False)
    #
    # x2 = x.clone().detach().requires_grad_(True)
    # tar_out = tar_fc(x2)
    #
    # diff_y = test_out - tar_out
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
    # target = torch.randn_like(test_out)
    # loss_1 = (target - test_out).pow(2).mean()
    # loss_2 = (target - tar_out).pow(2).mean()
    # loss_1.backward()
    # loss_2.backward()
    #
    # print(f'==================================================')
    # weight_grad_adda = test_fc.weight.grad
    # weight_grad_lsq = tar_fc.weight.grad
    # grad_diff = weight_grad_adda - weight_grad_lsq
    # grad_max_diff = torch.max(grad_diff.abs())
    # print(weight_grad_adda[0][0])
    # print(weight_grad_lsq[0][0])
    # print(f'weight_grad_max_diff_relative = {grad_max_diff / torch.max(weight_grad_lsq.abs()) * 100:.2f}%')
    # non_zero_count = torch.sum(grad_diff != 0).item()
    # print(f'weight_grad non_zero_count = {non_zero_count}')
    # print(f'==================================================')
    # x_grad_test = x.grad
    # x_grad_tar = x2.grad
    # close = torch.isclose(x_grad_test, x_grad_tar, atol = 1e-5)
    # num_not_close = close.numel() - close.sum().item()
    # proportion_not_close = num_not_close / close.numel()
    # print(f'x_grad_test = {x_grad_test[0]}')
    # print(f'x_grad_tar = {x_grad_tar[0]}')
    # print(f'ratio = {x_grad_test[0] / x_grad_tar[0]}')
    # print(f'proportion_not_close = {proportion_not_close}')
    # print(f'==================================================')
