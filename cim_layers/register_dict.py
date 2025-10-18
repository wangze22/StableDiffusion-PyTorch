# =============================== #
# @Author  : Wang Ze
# @Time    : 17:47
# @Software: PyCharm
# =============================== #
import torch.nn as nn

from cim_layers.layers_all import *
# ==================================== #
# nn 层
# ==================================== #
nn_layers = (nn.Conv2d, nn.Linear)

# ==================================== #
# QN 层
# ==================================== #
qn_linear_layers = (l_qn_lsq.Linear_qn_lsq,
                    l_q_lsq.Linear_q_lsq,
                    l_qn.Linear_quant_noise,
                    )

qn_conv_layers = (l_qn_lsq.Conv2d_qn_lsq,
                  l_q_lsq.Conv2d_q_lsq,
                  l_qn.Conv2d_quant_noise,
                  )
qn_layers = qn_linear_layers + qn_conv_layers
# ==================================== #
# 可以转换为 cim 的层，包含 CIM 仿真器运行时环境
# ==================================== #
cim_linear_layers = (
    # l_cim.Linear_cim,
    # l_cim_lsq.Linear_cim_LSQ,
    l_adda_cim.Linear_lsq_adda_cim,
l_adda_cim_opt.Linear_lsq_adda_cim,
    l_adda_multi_gains.Linear_lsq_adda_cim_multi_gains,
    l_adda_cim_row_split.Linear_lsq_adda_cim_row_split,
    l_144k.Linear_lsq_144k,
    l_144k_grad_test.Linear_lsq_144k,
)

cim_conv_layers = (
    # l_cim.Conv2d_cim,
    # l_cim_lsq.Conv2d_cim_LSQ,
    l_adda_cim.Conv2d_lsq_adda_cim,
l_adda_cim_opt.Conv2d_lsq_adda_cim,
    l_adda_multi_gains.Conv2d_lsq_adda_cim_multi_gains,
    l_adda_cim_row_split.Conv2d_lsq_adda_cim_row_split,
    l_144k.Conv2d_lsq_144k,
    l_144k_grad_test.Conv2d_lsq_144k,
)
cim_layers = cim_linear_layers + cim_conv_layers

# ==================================== #
# ADC 和 DAC 量化感知层
# ==================================== #
adda_linear_layers = (
    # l_adda.Linear_ADDA_aware,
    l_adda_lsq.Linear_lsq_adda,
    l_adda_cim.Linear_lsq_adda_cim,
    l_adda_multi_gains.Linear_lsq_adda_cim_multi_gains,
    l_adda_cim_row_split.Linear_lsq_adda_cim_row_split,
)
adda_conv_layers = (
    # l_adda.Conv2d_ADDA_aware,
    l_adda_cim.Conv2d_lsq_adda_cim,
    l_adda_lsq.Conv2d_lsq_adda,
    l_adda_multi_gains.Conv2d_lsq_adda_cim_multi_gains,
    l_adda_cim_row_split.Conv2d_lsq_adda_cim_row_split,
)
adda_layers = adda_conv_layers + adda_linear_layers

# ==================================== #
# DMAC 整数计算层
# ==================================== #
dmac_conv_layers = (l_lsq_int.Conv2d_lsq_int,)
dmac_linear_layers = (l_lsq_int.Linear_lsq_int,)
dmac_layers = dmac_conv_layers + dmac_linear_layers

# ==================================== #
# 片上计算层
# ==================================== #
chip_linear_layers = (
    # l_adda.Linear_ADDA_aware,
    l_144k.Linear_lsq_144k,
    l_512k.Linear_512k,
)
chip_conv_layers = (
    # l_adda.Conv2d_ADDA_aware,
    l_144k.Conv2d_lsq_144k,
    l_512k.Conv2d_512k,
)
chip_on_chip_layers = chip_conv_layers + chip_linear_layers

# ==================================== #
# 自定义层
# ==================================== #
custom_linear_layers = qn_linear_layers + cim_linear_layers + adda_linear_layers + chip_linear_layers

custom_conv_layers = qn_conv_layers + cim_conv_layers + adda_conv_layers + chip_conv_layers

custom_layers = custom_linear_layers + custom_conv_layers

# ==================================== #
# 卷积 + 全连接 层
# ==================================== #
linear_layers = custom_linear_layers + (nn.Linear,)
conv_layers = custom_conv_layers + (nn.Conv2d,)
op_layers = linear_layers + conv_layers

# ==================================== #
# 数字计算层
# ==================================== #
digital_compute_layers = ['enhance_layer', 'enhance_branch']
