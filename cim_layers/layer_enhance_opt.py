import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 原始实现（逐组循环版）
# -----------------------------
class GroupedLinearSlow(nn.Module):
    def __init__(self, in_features, out_features, groups, dtype=torch.float32, device=None):
        super().__init__()
        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_group = in_features // groups
        self.out_group = out_features // groups

        self.group_linears = nn.ModuleList([
            nn.Linear(self.in_group, self.out_group, device=device, dtype=dtype)
            for _ in range(groups)
        ])

        for layer in self.group_linears:
            layer.layer_flag = 'enhance_layer'
            layer.distance = 1

        with torch.no_grad():
            for layer in self.group_linears:
                layer.weight.copy_(torch.eye(self.out_group, self.in_group, dtype=dtype, device=device))
                layer.bias.fill_(0)

    def forward(self, x):
        # x: (..., in_features)
        chunks = x.split(self.in_group, dim=-1)
        out_chunks = [linear(chunk) for chunk, linear in zip(chunks, self.group_linears)]
        return torch.cat(out_chunks, dim=-1)

# -----------------------------
# 优化实现（爱因斯坦求和，无循环，接口完全一致）
# -----------------------------
class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups, dtype=torch.float32, device=None):
        super().__init__()
        assert in_features % groups == 0, "in_features must be divisible by groups"
        assert out_features % groups == 0, "out_features must be divisible by groups"

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.in_group = in_features // groups
        self.out_group = out_features // groups

        # 参数形状： (groups, out_group, in_group) 与 (groups, out_group)
        self.weight = nn.Parameter(torch.empty(groups, self.out_group, self.in_group, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(groups, self.out_group, device=device, dtype=dtype))

        # 按照原始实现进行初始化：每组权重为 eye(out_group, in_group)，bias=0
        with torch.no_grad():
            self.weight.zero_()
            # torch.eye(rows, cols) 生成 (out_group, in_group) 的“方阵截断对角”
            eye = torch.eye(self.out_group, self.in_group, device=device, dtype=dtype)
            self.weight.copy_(eye.expand(self.groups, -1, -1))
            self.bias.zero_()

    def forward(self, x):
        # x: (..., in_features)
        # 先 reshape 成 (..., groups, in_group)
        x_shape = x.shape
        assert x_shape[-1] == self.in_features, "Invalid input last-dim size"

        xg = x.view(*x_shape[:-1], self.groups, self.in_group)  # (..., G, I_g)

        # einsum：(..., G, I_g) * (G, O_g, I_g)^T[在公式中对应goi] -> (..., G, O_g)
        # 使用 '...gi,goi->...go'，其中:
        #   ...gi: 输入 (..., groups, in_group)
        #   goi : 权重 (groups, out_group, in_group)
        #   结果 (..., groups, out_group)
        y = torch.einsum('...gi,goi->...go', xg, self.weight) + self.bias  # 广播加法

        # 拼回 (..., out_features)
        y = y.reshape(*x_shape[:-1], self.out_features)
        return y

# -----------------------------
# 工具函数
# -----------------------------
def copy_params_from_slow_to_fast(slow: GroupedLinearSlow, fast: GroupedLinear):
    """将慢版每组的 Linear 参数拷贝到快版的 weight/bias（用于严格一致性对比）"""
    with torch.no_grad():
        for g, layer in enumerate(slow.group_linears):
            fast.weight[g].copy_(layer.weight)
            fast.bias[g].copy_(layer.bias)

def rand_fill_slow(slow: GroupedLinearSlow):
    """给慢版参数随机赋值（非对称），然后可拷贝到快版做一致性测试"""
    with torch.no_grad():
        for layer in slow.group_linears:
            layer.weight.normal_(mean=0.0, std=0.5)
            layer.bias.uniform_(-0.1, 0.1)

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def benchmark_forward(module, x, warmup=5, iters=50):
    # 小型基准测试，支持 CPU/CUDA
    if x.is_cuda:
        torch.cuda.synchronize()
    # 预热
    for _ in range(warmup):
        y = module(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = module(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters

def grad_check(module, x):
    """简单的梯度一致性检查：对 sum(output) 做 backward"""
    module.zero_grad(set_to_none=True)
    y = module(x).sum()
    y.backward()
    # 拼接并返回所有参数梯度用于比较
    grads = []
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads) if grads else torch.tensor([])

# -----------------------------
# 主测试
# -----------------------------
def run_tests(device):
    print(f"\n===== Device: {device} =====")

    # 配置
    torch.manual_seed(1234)
    dtype = torch.float32

    # 多组配置测试（覆盖非等方阵）
    configs = [
        (1024, 1024, 32),   # 方阵
        (768,  1024, 32),   # 列 < 行
        (1024, 768,  32),   # 行 < 列
        (512,  512,  16),   # 中等
    ]

    for (in_f, out_f, G) in configs:
        print(f"\n-- Config: in={in_f}, out={out_f}, groups={G}")

        # 准备模块（慢/快）
        slow = GroupedLinearSlow(in_f, out_f, G, dtype=dtype, device=device)
        fast = GroupedLinear(in_f, out_f, G, dtype=dtype, device=device)

        # 先随机化 slow，再把参数拷贝到 fast，确保两边权重/偏置完全一致
        rand_fill_slow(slow)
        copy_params_from_slow_to_fast(slow, fast)

        # 输入两种形状：二维 (B, F_in) 与 三维 (B, T, F_in)
        B, T = 128, 17
        x2 = torch.randn(B, in_f, dtype=dtype, device=device)
        x3 = torch.randn(B, T, in_f, dtype=dtype, device=device)

        # ---- 前向一致性
        with torch.no_grad():
            y2_s = slow(x2); y2_f = fast(x2)
            y3_s = slow(x3); y3_f = fast(x3)
        diff2 = max_abs_diff(y2_s, y2_f)
        diff3 = max_abs_diff(y3_s, y3_f)
        print(f"Forward max diff (2D): {diff2:.3e} | (3D): {diff3:.3e}")

        # ---- 反向梯度一致性（对 sum 输出求导）
        g2_s = grad_check(slow, x2.clone().requires_grad_(True))
        g2_f = grad_check(fast, x2.clone().requires_grad_(True))
        gdiff = max_abs_diff(g2_s, g2_f)
        print(f"Grad max diff (params, 2D): {gdiff:.3e}")

        # ---- 简易性能测试（只测 2D，避免耗时过长）
        t_s = benchmark_forward(slow, x2, warmup=5, iters=50)
        t_f = benchmark_forward(fast, x2, warmup=5, iters=50)
        print(f"Speed (avg per iter, ms): slow={t_s*1e3:.3f} | fast={t_f*1e3:.3f} | speedup={t_s/max(t_f,1e-12):.2f}x")

if __name__ == "__main__":
    # CPU
    run_tests(device="cpu")

    # CUDA（如可用）
    if torch.cuda.is_available():
        run_tests(device="cuda")
