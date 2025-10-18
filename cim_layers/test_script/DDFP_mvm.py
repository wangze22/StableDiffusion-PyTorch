import torch

from cim_layers.layers_utils_lsq import *

d_rows = 100
d_cols = 10
w_cols = 20


def unique_values(tensor):
    unique_values = torch.unique(tensor)
    return unique_values.numel()  # 返回唯一值的数量


# =================== #
# 输入量化
# =================== #
x = torch.randn(d_rows, d_cols, requires_grad = True)
x_q, x_s = data_quant_pass(x, data_bit = 4, isint = 1)
# =================== #
# 权重量化
# =================== #
w = torch.randn(d_rows, w_cols, requires_grad = True)
w_q, w_s = data_quant_pass(w, data_bit = 4, isint = 1)
w_q.retain_grad()
# =================== #
# 输出量化
# =================== #
y = torch.matmul(x_q.transpose(1, 0), w_q)
y.retain_grad()
y_q, y_s = data_quant_pass(y, data_bit = 4, isint = 1)
y_q.retain_grad()

# =================== #
# 求 Loss
# =================== #
tar = (torch.randn_like(y) * y_q.std()).round()
loss = ((tar - y_q) ** 2).sum()
loss.backward()

# =================== #
# dL/d(y_q)
# =================== #
grad_L_yq = 2 * (y_q - tar)
print(y_q.grad - grad_L_yq)
close = torch.isclose(y_q.grad, grad_L_yq, atol = 1e-5)
num_not_close = close.numel() - close.sum().item()
proportion_not_close = num_not_close / close.numel()
print(proportion_not_close)

# =================== #
# dL/d(y)
# =================== #
grad_yq_y = y_s
grad_L_y = grad_yq_y * grad_L_yq
print(y.grad - grad_L_y)
close = torch.isclose(y.grad, grad_L_y, atol = 1e-5)
num_not_close = close.numel() - close.sum().item()
proportion_not_close = num_not_close / close.numel()
print(proportion_not_close)

# =================== #
# dL/d(w_q)
# =================== #
# 这一步需要使用到还原单元
grad_L_wq = torch.matmul(x_q, grad_L_yq) * y_s
print(w_q.grad - grad_L_wq)
close = torch.isclose(w_q.grad, grad_L_wq, atol = 1e-5)
num_not_close = close.numel() - close.sum().item()
proportion_not_close = num_not_close / close.numel()
print(proportion_not_close)

# ====================================== #
# 方案一：直接在量化后的权重上进行基于整数的更新
# ====================================== #
# learn_rate 采用移位的方式
learn_rate = 2**-6
# 在没有经过 y_s 系数还原之前，先对 w_q 的梯度进行 lr 移位
grad_L_wq_lr = (torch.matmul(x_q, grad_L_yq) * learn_rate).floor()
# 进行 y_s 系数还原，得到实际的梯度*学习率
grad_L_wq_lr_q = (grad_L_wq_lr * y_s).round()
# 对权重进行更新
w_q_new = w_q + grad_L_wq_lr_q

# ====================================== #
# 方案二：对权重进行还原，然后更新梯度
# ====================================== #
# 这个方法避免了对梯度进行四舍五入带来的误差，但是需要使用 FP 加法器和乘法器
w_new = w_q / w_s
w_new += grad_L_wq * learn_rate / w_s