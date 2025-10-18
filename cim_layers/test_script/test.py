import torch
import torch.nn as nn

# 创建一个随机的输入张量，假设是一个单通道图像
input_tensor = torch.randn(1, 1, 10, 10)

# 创建 Unfold 操作，kernel_size=3, padding=1, stride=1
unfold = nn.Unfold(kernel_size=3, padding=1, stride=1)

# 使用 Unfold 展开张量
unfolded_tensor = unfold(input_tensor)
print("Unfolded shape:", unfolded_tensor.shape)

# 创建 Fold 操作，输出大小应与原始输入相同
fold = nn.Fold(output_size=(10, 10), kernel_size=3, padding=1, stride=1)

# 使用 Fold 恢复张量
folded_tensor = fold(unfolded_tensor)
print("Folded shape:", folded_tensor.shape)

# 检查 Fold 后的张量是否与原始输入相近
print("Restored tensor is close to the original:", torch.allclose(input_tensor, folded_tensor))
