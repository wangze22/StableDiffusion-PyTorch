import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import math
from itertools import product


def _divisors(x: int):
    """Return all positive divisors of x (sorted)."""
    divs = set()
    for d in range(1, int(x ** 0.5) + 1):
        if x % d == 0:
            divs.add(d)
            divs.add(x // d)
    return sorted(divs)


def _closest_divisor(val, candidates):
    """Return the candidate divisor closest to val."""
    return min(candidates, key = lambda d: abs(d - val))


def ideal_hidden_channels(C_in: int,
                          C_out: int,
                          k_h: int,
                          k_w: int,
                          ops_factor: float) -> int:
    """
    返回使 LoRA(两层, 不分组) 的计算量
      = (ops_factor)^(1/3) × 原始卷积计算量
    的最接近整数 hidden_channels。
    """
    r = ops_factor ** (1 / 3)  # 目标三分之一压缩比例
    h_float = r * C_in * C_out * (k_h * k_w) / (C_in + C_out)
    h_int = max(1, round(h_float))  # 取最近整数, 至少为 1
    return h_int


def ideal_hidden_features(F_in: int, F_out: int, ops_factor: float) -> float:
    """
    在 **不分组** 情况下，使
        (C_in*h + h*C_out) / (C_in*C_out)  =  (ops_factor)^(1/3)
    的解析 h（浮点）。
    """
    r = ops_factor ** (1 / 3)  # 每一维应压缩比例
    return (r * F_in * F_out) / (F_in + F_out)


def analyze_lora_conv_ops(in_channels: int,
                          out_channels: int,
                          kh: int,
                          kw: int,
                          ops_factor: float):
    """
    协同搜索 (group1, group2, hidden_channels)，
    让 LoRA-Conv2d 两层总计算量 ≈ 原始卷积 * ops_factor，
    并让三种压缩手段贡献尽量均衡。
    返回 dict:
        - group1, group2
        - hidden_channels
        - estimated_ops_factor
    """
    ops_orig = in_channels * out_channels * kh * kw

    # 目标让 rank/g1/g2 各压缩一个 base_ratio
    base_ratio = (ops_factor) ** (1 / 3)  # ≈ 需要的每项比例
    hid_ideal = ideal_hidden_channels(in_channels, out_channels, kh, kw, ops_factor)

    hid_candidates = range(max(1, int(hid_ideal * 0.1)),
                           max(1, int(hid_ideal * 10)) + 1)

    div_in = _divisors(in_channels)
    div_out = _divisors(out_channels)

    best_cfg = None
    best_loss1 = float('inf')
    best_loss2 = float('inf')

    for hc in hid_candidates:
        div_hid = _divisors(hc)
        # 理想的 group ≈ 1/base_ratio
        g1_ideal = g2_ideal = 1 / base_ratio

        # 找到 hid 中可用且能整除 in/out 的最佳 g1/g2
        g1 = _closest_divisor(g1_ideal, [d for d in div_hid if d in div_in])
        g2 = _closest_divisor(g2_ideal, [d for d in div_hid if d in div_out])

        # 计算两层 LoRA ops（省略 H×W，因为比值抵消）
        ops1 = in_channels * hc // g1
        ops2 = hc * out_channels // g2
        lora_ops = ops1 + ops2
        factor = lora_ops / ops_orig

        # ------ 1) 目标误差 ------
        loss1 = abs(factor - ops_factor)

        # ------ 2) 细分三种压缩比例 ------
        # 2-a rank  (g1=g2=1, 使用当前 h)
        r_rank = (in_channels * hc + hc * out_channels) / ops_orig

        # 2-b g1    (h = hid_ideal, g1 = 当前 g1, g2=1)
        r_g1 = (in_channels * hid_ideal / g1 + hid_ideal * out_channels) / ops_orig

        # 2-c g2    (h = hid_ideal, g1=1, g2 = 当前 g2)
        r_g2 = (in_channels * hid_ideal + hid_ideal * out_channels / g2) / ops_orig

        # L2 距离：三者偏离 base_ratio 的总合
        loss2 = math.sqrt((r_rank - base_ratio) ** 2 +
                          (r_g1 - base_ratio) ** 2 +
                          (r_g2 - base_ratio) ** 2)

        if (loss1 < best_loss1) or (loss1 == best_loss1 and loss2 < best_loss2):
            best_loss1, best_loss2 = loss1, loss2
            best_cfg = dict(group1 = g1,
                            group2 = g2,
                            hidden_channels = hc,  # 或 hidden_features=h
                            estimated_ops_factor = factor)

    # fallback（极端情况下）
    if best_cfg is None:
        hc = max(1, int(hid_ideal))
        factor = (in_channels * hc + hc * out_channels) / ops_orig
        best_cfg = dict(group1 = 1, group2 = 1,
                        hidden_channels = hc,
                        estimated_ops_factor = factor)

    return best_cfg


def analyze_lora_linear_ops(in_features: int,
                            out_features: int,
                            ops_factor: float):
    """
    协同搜索 (group1, group2, hidden_features) 使
        (in*h/g1 + h*out/g2) / (in*out) ≈ ops_factor
    同时让 rank、g1、g2 三项压缩幅度基本均衡。
    """
    original_ops = in_features * out_features
    base_ratio = ops_factor ** (1 / 3)

    # 1) 解析求得 rank 压缩应取的理想 hidden (浮点)
    hid_ideal = ideal_hidden_features(in_features, out_features, ops_factor)

    # 2) 枚举整数 hidden（±30% 搜索）
    hid_candidates = range(max(1, int(hid_ideal * 0.1)),
                           max(1, int(hid_ideal * 10)) + 1)

    div_in = _divisors(in_features)
    div_out = _divisors(out_features)

    best_cfg = None
    best_loss1 = float('inf')
    best_loss2 = float('inf')

    for h in hid_candidates:
        div_h = _divisors(h)

        # 目标 group ≈ 1/base_ratio
        g_ideal = 1 / base_ratio
        g1 = _closest_divisor(g_ideal, [d for d in div_h if d in div_in])
        g2 = _closest_divisor(g_ideal, [d for d in div_h if d in div_out])

        # 重新计算 LoRA ops
        ops1 = in_features * h // g1
        ops2 = h * out_features // g2
        lora_ops = ops1 + ops2
        factor = lora_ops / original_ops

        # ------ 1) 目标误差 ------
        loss1 = abs(factor - ops_factor)

        # ------ 2) 细分三种压缩比例 ------
        # 2-a rank  (g1=g2=1, 使用当前 h)
        r_rank = (in_features * h + h * out_features) / original_ops

        # 2-b g1    (h = hid_ideal, g1 = 当前 g1, g2=1)
        r_g1 = (in_features * hid_ideal / g1 + hid_ideal * out_features) / original_ops

        # 2-c g2    (h = hid_ideal, g1=1, g2 = 当前 g2)
        r_g2 = (in_features * hid_ideal + hid_ideal * out_features / g2) / original_ops

        # L2 距离：三者偏离 base_ratio 的总合
        loss2 = math.sqrt((r_rank - base_ratio) ** 2 +
                          (r_g1 - base_ratio) ** 2 +
                          (r_g2 - base_ratio) ** 2)


        if (loss1 < best_loss1) or (loss1 == best_loss1 and loss2 < best_loss2):
            best_loss1, best_loss2 = loss1, loss2
            best_cfg = dict(group1 = g1,
                            group2 = g2,
                            hidden_features = h,
                            estimated_ops_factor = factor)

    # fallback（极端整除失败）
    if best_cfg is None:
        h = max(1, round(hid_ideal))
        factor = (in_features * h + h * out_features) / original_ops
        best_cfg = dict(group1 = 1, group2 = 1,
                        hidden_features = h,
                        estimated_ops_factor = factor)

    return best_cfg


def analyze_enhance_conv_ops(in_channels, out_channels, kh, kw, ops_factor):
    """
    分析用于 EnhanceLayerConv2d 的最佳 groups 值。

    参数：
        - in_channels: 输入通道数 = 输出通道数（原始卷积）
        - kh, kw: 原始卷积核大小
        - ops_factor: 增强层计算量占原始层的比例目标

    返回：
        - group: 最佳分组数（使得计算量不超过目标）
        - estimated_ops_factor: 实际计算量占比
    """
    original_ops = in_channels * out_channels * kh * kw  # 原始卷积的计算量（忽略 H×W）

    # 向上搜索最小满足 ops_factor 的合法 group（能整除）
    for g in range(1, out_channels + 1):
        if out_channels % g != 0:
            continue
        enhance_ops = out_channels * out_channels // g  # 1×1 分组卷积计算量
        ratio = enhance_ops / original_ops
        if ratio <= ops_factor:
            return {"group": g, "estimated_ops_factor": ratio}

    # fallback：找不到满足 ops_factor 的，返回最大 group（depthwise）
    return {"group": out_channels, "estimated_ops_factor": 1.0 / (kh * kw)}


def analyze_enhance_linear_ops(in_features, out_features, ops_factor):
    """
    分析用于 EnhanceLayerLinear 的最佳 groups 值。

    参数：
        - out_features: 原始 Linear 的输出通道数（等于增强层输入/输出）
        - ops_factor: 增强层计算量占原始 Linear 的比例目标

    返回：
        - group: 建议的分组数
        - estimated_ops_factor: 实际计算量占比
    """
    original_ops = in_features * out_features  # 原始 FC 层的计算量

    for g in range(1, out_features + 1):
        if out_features % g != 0:
            continue
        enhance_ops = out_features * out_features // g
        ratio = enhance_ops / original_ops
        if ratio <= ops_factor:
            return {"group": g, "estimated_ops_factor": ratio}

    # fallback（找不到满足 ops_factor 的）
    return {"group": out_features, "estimated_ops_factor": 1.0}


class GroupedLinearConv1d(nn.Module):
    def __init__(self, in_features, out_features, groups, dtype = torch.float32, device = None):
        super().__init__()
        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_group = in_features // groups
        self.out_group = out_features // groups

        # 构建 group conv，等效于 grouped linear
        self.group_conv = nn.Conv1d(
            in_channels = in_features,
            out_channels = out_features,
            kernel_size = 1,
            groups = groups,
            bias = True,
            dtype = dtype,
            device = device
        )

        # 设置标记
        self.group_conv.layer_flag = 'enhance_layer'
        self.group_conv.distance = 1

        # 初始化为 group-wise 单位映射（仅当 in_group == out_group 时可行）
        if self.in_group == self.out_group:
            with torch.no_grad():
                weight = torch.zeros(out_features, self.in_group, 1, device = device, dtype = dtype)
                eye = torch.eye(self.in_group, dtype = dtype, device = device).unsqueeze(-1)
                for g in range(groups):
                    start = g * self.out_group
                    weight[start:start + self.out_group] = eye
                self.group_conv.weight.copy_(weight)
                self.group_conv.bias.zero_()
        else:
            # fallback: 使用标准初始化
            nn.init.kaiming_uniform_(self.group_conv.weight, a = math.sqrt(5))
            if self.group_conv.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.group_conv.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.group_conv.bias, -bound, bound)

    def forward(self, x):
        if x.dim() == 2:
            # (B, in_features) -> (B, in_features, 1)
            x = x.unsqueeze(-1)
            out = self.group_conv(x)  # (B, out_features, 1)
            return out.squeeze(-1)  # (B, out_features)
        elif x.dim() == 3:
            # (B, T, in_features) -> (B, in_features, T)
            x = x.permute(0, 2, 1)
            out = self.group_conv(x)  # (B, out_features, T)
            return out.permute(0, 2, 1)  # -> (B, T, out_features)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")


class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups, dtype = torch.float32, device = None):
        super().__init__()
        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_group = in_features // groups
        self.out_group = out_features // groups

        self.group_linears = nn.ModuleList([
            nn.Linear(self.in_group, self.out_group, device = device, dtype = dtype)
            for _ in range(groups)
        ])

        for layer in self.group_linears:
            layer.layer_flag = 'enhance_layer'
            layer.distance = 1

        with torch.no_grad():
            for layer in self.group_linears:
                layer.weight.copy_(torch.eye(self.out_group, self.in_group, dtype = dtype, device = device))
                layer.bias.fill_(0)

    def forward(self, x):
        chunks = x.split(self.in_group, dim = -1)
        out_chunks = [linear(chunk) for chunk, linear in zip(chunks, self.group_linears)]
        return torch.cat(out_chunks, dim = -1)


class EnhanceLayerConv2d(nn.Module):
    def __init__(self, original_conv, ops_factor = 0.05):
        super().__init__()
        self.original_conv = original_conv
        out_channels = self.original_conv.out_channels
        in_channels = original_conv.in_channels
        kh, kw = original_conv.kernel_size

        weight = original_conv.weight
        dtype = weight.dtype
        device = weight.device

        # ---- 自动分析最佳 groups ----
        cfg = analyze_enhance_conv_ops(in_channels, out_channels, kh, kw, ops_factor)
        groups = cfg["group"]
        self.groups = groups

        self.conv_enhance = nn.Conv2d(
            out_channels, out_channels,
            kernel_size = 1, stride = 1,
            padding = 0, groups = groups,
            device = device, dtype = dtype
        )

        self.conv_enhance.layer_flag = 'enhance_layer'
        self.conv_enhance.distance = 1

        self._initialize_weights()

        print(f"[Enhance-Conv] ops_factor={ops_factor:.3f} → estimated={cfg['estimated_ops_factor']:.4f}, group={groups}")

    def _initialize_weights(self):
        eye_matrix = torch.eye(self.conv_enhance.out_channels // self.groups,
                               self.conv_enhance.out_channels // self.groups).repeat(self.groups, 1, 1)
        with torch.no_grad():
            self.conv_enhance.weight.copy_(eye_matrix.view(self.conv_enhance.out_channels,
                                                           self.conv_enhance.out_channels // self.groups, 1, 1))
            self.conv_enhance.bias.fill_(0)

    def forward(self, x):
        x = self.original_conv(x)
        x = self.conv_enhance(x)
        return x


class EnhanceLayerLinear(nn.Module):
    def __init__(self, original_linear, ops_factor = 0.05):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 获取原始 linear 层的 device 和 dtype
        weight = original_linear.weight
        dtype = weight.dtype
        device = weight.device

        # 自动分析最佳 group 数
        cfg = analyze_enhance_linear_ops(in_features, out_features, ops_factor)
        groups = cfg["group"]

        self.fc1 = GroupedLinear(out_features, out_features, groups, dtype = dtype, device = device)

        print(f"[Enhance-Linear] ops_factor={ops_factor:.3f} → estimated={cfg['estimated_ops_factor']:.4f}, group={groups}")

    def forward(self, x):
        x = self.original_linear(x)
        x = self.fc1(x)
        return x


class EnhanceBranchConv2d_LoR(nn.Module):
    def __init__(self, original_conv, ops_factor = 0.05, relu = False, sigmoid = True):
        super().__init__()
        self.original_conv = original_conv
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kh, kw = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.relu = relu
        self.sigmoid = sigmoid

        weight = original_conv.weight
        dtype = weight.dtype
        device = weight.device

        # --- 自动分析 LoRA 配置 ---
        cfg = analyze_lora_conv_ops(in_channels, out_channels, kh, kw, ops_factor)
        hidden_channels = cfg["hidden_channels"]
        group1 = cfg["group1"]
        group2 = cfg["group2"]
        estimated = cfg["estimated_ops_factor"]

        # --- 构建两个 Conv2d LoRA 层 ---
        self.enhance_branch_1 = nn.Conv2d(
            in_channels, hidden_channels,
            kernel_size = 1, stride = 1, padding = self.padding,
            groups = group1, dtype = dtype, device = device
        )

        self.enhance_branch_2 = nn.Conv2d(
            hidden_channels, out_channels,
            kernel_size = 1, stride = 1, padding = self.padding,
            groups = group2, dtype = dtype, device = device
        )

        print(f"[LoRA-Conv2d] ops_factor={ops_factor:.3f} → estimated={estimated:.4f}, "
              f"group1={group1}, group2={group2}, "
              f"hidden={hidden_channels}")

        # self.layer_flag = 'enhance_branch'
        self.enhance_branch_1.layer_flag = 'enhance_branch'
        self.enhance_branch_2.layer_flag = 'enhance_branch'
        # 如果 stride 不是 1，定义池化层来调整输入尺寸
        if any(s != 1 for s in self.stride):
            self.pool = nn.AvgPool2d(kernel_size = self.stride, stride = self.stride, padding = 0)
        else:
            self.pool = None
        if self.sigmoid:
            self.blend_factor = nn.Parameter(torch.tensor(-6.9068))
        else:
            self.blend_factor = nn.Parameter(torch.tensor(0.0))
        self.std_ori = None
        self.mean_ori = None
        self.std_enhance = None
        self.mean_enhance = None

    def merge_feature_map(self, original_output, enhance_output):
        if self.sigmoid:
            combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                              + enhance_output * torch.sigmoid(self.blend_factor)
        else:
            combined_output = original_output * (1 - self.blend_factor) \
                              + enhance_output * self.blend_factor
        self.std_ori = original_output.std()
        self.mean_ori = original_output.mean()
        self.std_enhance = enhance_output.std()
        self.mean_enhance = enhance_output.mean()
        return combined_output

    def forward(self, x):
        original_output = self.original_conv(x)
        enhance_output = self.enhance_branch_1(x)
        if self.relu:
            enhance_output = F.relu(enhance_output)
        enhance_output = self.enhance_branch_2(enhance_output)
        # 使用池化层调整输入尺寸
        if self.pool is not None:
            enhance_output = self.pool(enhance_output)

        # 调整 enhance_output 以匹配 original_output
        diff_y = original_output.shape[2] - enhance_output.shape[2]
        diff_x = original_output.shape[3] - enhance_output.shape[3]

        # 如果尺寸不同，执行填充或裁剪
        if diff_y != 0 or diff_x != 0:
            if diff_y < 0 or diff_x < 0:
                enhance_output = enhance_output[:, :, :original_output.shape[2], :original_output.shape[3]]

            elif diff_y > 0 or diff_x > 0:
                pad = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
                enhance_output = nn.functional.pad(enhance_output, pad)

        combined_output = self.merge_feature_map(original_output, enhance_output)
        return combined_output


# class EnhanceLayerLinear(nn.Module):
#     def __init__(self, original_linear):
#         super().__init__()
#         self.original_linear = original_linear
#         out_features = original_linear.out_features
#         self.fc1 = nn.Linear(out_features, out_features)
#         self.fc1.layer_flag = 'enhance_layer'
#         self.fc1.distance = 1
#         # self.layer_flag = 'enhance_layer'
#         # 初始化 enhance_branch_1 的权重为单位矩阵，偏置为 0
#         with torch.no_grad():
#             self.fc1.weight.copy_(torch.eye(out_features))
#             self.fc1.bias.fill_(0)
#
#     def forward(self, x):
#         x = self.original_linear(x)
#         x = self.fc1(x)
#         return x


class EnhanceBranchLinear_LoR(nn.Module):
    def __init__(self, original_linear, ops_factor = 0.05, relu = False, sigmoid = True):
        super().__init__()
        self.original_linear = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.relu = relu
        self.sigmoid = sigmoid

        # 获取 weight 的 dtype 和 device
        weight = original_linear.weight
        dtype = weight.dtype
        device = weight.device

        # 自动分析最佳 groups 和 hidden size
        cfg = analyze_lora_linear_ops(in_features, out_features, ops_factor)
        hidden_features = cfg["hidden_features"]
        group1 = cfg["group1"]
        group2 = cfg["group2"]
        estimated = cfg["estimated_ops_factor"]

        # 构建两层 GroupedLinear
        self.enhance_branch_1 = GroupedLinear(in_features, hidden_features, group1, dtype = dtype, device = device)
        self.enhance_branch_2 = GroupedLinear(hidden_features, out_features, group2, dtype = dtype, device = device)

        print(f"[LoRA-Linear] ops_factor={ops_factor:.3f} → estimated={estimated:.4f}, "
              f"group1={group1}, group2={group2}, "
              f"hidden_features={hidden_features}")

        if self.sigmoid:
            self.blend_factor = nn.Parameter(torch.tensor(-6.9068))
        else:
            self.blend_factor = nn.Parameter(torch.tensor(0.0))

        # self.layer_flag = 'enhance_branch'
        self.enhance_branch_1.layer_flag = 'enhance_branch'
        self.enhance_branch_1.distance = 1
        self.enhance_branch_2.layer_flag = 'enhance_branch'
        self.enhance_branch_2.distance = 1

        self.std_ori = None
        self.mean_ori = None
        self.std_enhance = None
        self.mean_enhance = None

    def merge_feature_map(self, original_output, enhance_output):
        if self.sigmoid:
            combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
                              + enhance_output * torch.sigmoid(self.blend_factor)
        else:
            combined_output = original_output * (1 - self.blend_factor) \
                              + enhance_output * self.blend_factor
        self.std_ori = original_output.std()
        self.mean_ori = original_output.mean()
        self.std_enhance = enhance_output.std()
        self.mean_enhance = enhance_output.mean()
        return combined_output

    def forward(self, x):
        original_output = self.original_linear(x)
        enhanced_output = self.enhance_branch_1(x)
        if self.relu:
            enhanced_output = F.relu(enhanced_output)
        enhanced_output = self.enhance_branch_2(enhanced_output)
        combined_output = self.merge_feature_map(original_output, enhanced_output)
        return combined_output

# =============================================== #
# 弃用函数
# =============================================== #
# class EnhanceBranchConv2d(nn.Module):
#     def __init__(self, original_conv, groups = 1):
#         super().__init__()
#         self.original_conv = original_conv
#         in_channels = self.original_conv.in_channels
#         out_channels = self.original_conv.out_channels
#         self.stride = self.original_conv.stride
#         self.padding = self.original_conv.padding
#
#         # 定义 1x1 的卷积层作为增强层
#         self.enhance_layer = nn.Conv2d(in_channels, out_channels,
#                                        kernel_size = 1, stride = 1, padding = self.padding,
#                                        groups = groups)
#         # self.layer_flag = 'enhance_branch'
#         self.enhance_layer.layer_flag = 'enhance_branch'
#         # 如果 stride 不是 1，定义池化层来调整输入尺寸
#         if any(s != 1 for s in self.stride):
#             self.pool = nn.AvgPool2d(kernel_size = self.stride, stride = self.stride, padding = 0)
#         else:
#             self.pool = None
#
#         self.blend_factor = nn.Parameter(torch.tensor(0.0))
#
#     def merge_feature_map(self, original_output, enhance_output):
#         combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
#                           + enhance_output * torch.sigmoid(self.blend_factor)
#         return combined_output
#
#     def forward(self, x):
#         original_output = self.original_conv(x)
#         enhance_output = self.enhance_layer(x)
#         # 使用池化层调整输入尺寸
#         if self.pool is not None:
#             enhance_output = self.pool(enhance_output)
#
#         # 调整 enhance_output 以匹配 original_output
#         diff_y = original_output.shape[2] - enhance_output.shape[2]
#         diff_x = original_output.shape[3] - enhance_output.shape[3]
#
#         if diff_y < 0 or diff_x < 0:
#             enhance_output = enhance_output[:, :, :original_output.shape[2], :original_output.shape[3]]
#         elif diff_y > 0 or diff_x > 0:
#             pad = (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
#             enhance_output = nn.functional.pad(enhance_output, pad)
#
#         combined_output = self.merge_feature_map(original_output, enhance_output)
#         return combined_output

# 在全连接层后面加入LoR结构，有可能会让原来的高维信息丢失，因此不使用该函数
# class EnhanceLayerLinear_LoR(nn.Module):
#     def __init__(self, original_linear, rank_factor = 0.25):
#         super().__init__()
#         self.original_linear = original_linear
#         in_features = original_linear.in_features
#         out_features = original_linear.out_features
#         hidden_size = max(int(max(in_features, out_features) * rank_factor), 1)
#         self.fc1 = nn.Linear(out_features, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, out_features)
#         self.layer_flag = 'enhance_layer'
#         self.fc1.layer_flag = 'enhance_layer'
#         self.fc2.layer_flag = 'enhance_layer'
#
#     def forward(self, x):
#         x = self.original_linear(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
#
#
# class InterpolateBranchLinear(nn.Module):
#     def __init__(self, original_linear):
#         super().__init__()
#         self.original_linear = original_linear
#         self.blend_factor = nn.Parameter(torch.tensor(0.0))
#
#     def merge_feature_map(self, original_output, enhance_output):
#         combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
#                           + enhance_output * torch.sigmoid(self.blend_factor)
#         return combined_output
#
#     def forward(self, x):
#         original_output = self.original_linear(x)
#
#         # 调整 enhance_output 以匹配 original_output 的形状
#         enhanced_output = torch.nn.functional.interpolate(x.unsqueeze(1),
#                                                           size = original_output.shape[-1],
#                                                           mode = 'linear',
#                                                           align_corners = False).squeeze(1)
#
#         combined_output = self.merge_feature_map(original_output, enhanced_output)
#         return combined_output
#
#
# # 该函数目前有问题，interpolate只能对宽高进行插值，而无法改变通道数
# # GPT目前的解决方案就是通过1x1的卷积调整通道数
# # 因此单纯依靠interpolate可能无法实现增强分支
# class InterpolateBranchConv2d(nn.Module):
#     def __init__(self, original_conv):
#         super().__init__()
#         self.original_conv = original_conv
#         self.stride = self.original_conv.stride
#         self.padding = self.original_conv.padding
#
#         self.blend_factor = nn.Parameter(torch.tensor(0.0))
#
#     def merge_feature_map(self, original_output, enhance_output):
#         combined_output = original_output * (1 - torch.sigmoid(self.blend_factor)) \
#                           + enhance_output * torch.sigmoid(self.blend_factor)
#         return combined_output
#
#     def forward(self, x):
#         original_output = self.original_conv(x)
#
#         # 调整 enhance_output 以匹配 original_output
#         enhance_output = torch.nn.functional.interpolate(x,
#                                                          size = original_output.shape[2:],
#                                                          mode = 'bilinear', align_corners = False)
#
#         combined_output = self.merge_feature_map(original_output, enhance_output)
#         return combined_output
