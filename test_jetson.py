#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, torch.nn.functional as F, time, traceback

# ------- 固定参数（跟你出错形状一致，可自行改动） -------
BATCH_SIZE   = 2
IN_CHANNELS  = 96
OUT_CHANNELS = 48
H, W         = 256, 256
KERNEL_SIZE  = (3, 3)
STRIDE       = (1, 1)
PADDING      = (1, 1)
DILATION     = (1, 1)
GROUPS       = 1
DEVICE       = "cuda:0"
# -------------------------------------------------------

print("="*80)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("Device:", torch.cuda.get_device_name(0))
free,total = torch.cuda.mem_get_info()
print(f"Mem free/total: {free//1024//1024} MB / {total//1024//1024} MB")
print("="*80)

dev = torch.device(DEVICE)
torch.cuda.set_device(dev)

def run(title, fn):
    print(f"\n--- {title} ---")
    try:
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize()
        print(f"{title}: OK in {(time.time()-t0)*1000:.2f} ms; out={tuple(out.shape) if hasattr(out,'shape') else type(out)}")
    except Exception as e:
        print(f"{title}: FAIL")
        print(traceback.format_exc())

# 1) CUDA/GPU 是否正常（用 GEMM 测一下）
def test_cuda_gemm():
    a = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=dev, dtype=torch.float32)
    return a @ b

# 2) CPU 上的 conv2d（同形状），排除形状不合法
def test_cpu_conv():
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, H, W, device="cpu", dtype=torch.float32)
    w = torch.randn(OUT_CHANNELS, IN_CHANNELS // GROUPS, *KERNEL_SIZE, device="cpu", dtype=torch.float32)
    b = torch.randn(OUT_CHANNELS, device="cpu", dtype=torch.float32)
    return F.conv2d(x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS)

# 3) GPU + cuDNN（你现在报错的路径）
def test_gpu_cudnn_conv_fp32():
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, H, W, device=dev, dtype=torch.float32)
    w = torch.randn(OUT_CHANNELS, IN_CHANNELS // GROUPS, *KERNEL_SIZE, device=dev, dtype=torch.float32)
    b = torch.randn(OUT_CHANNELS, device=dev, dtype=torch.float32)
    return F.conv2d(x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS)

# 4) GPU + 非 cuDNN + FP32（验证是否“只有 cuDNN 挂”）
def test_gpu_nocudnn_conv_fp32():
    from torch.backends.cudnn import flags as cudnn_flags
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, H, W, device=dev, dtype=torch.float32).contiguous()
    w = torch.randn(OUT_CHANNELS, IN_CHANNELS // GROUPS, *KERNEL_SIZE, device=dev, dtype=torch.float32).contiguous()
    b = torch.randn(OUT_CHANNELS, device=dev, dtype=torch.float32).contiguous()
    with cudnn_flags(enabled=False):
        return F.conv2d(x, w, b, stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS)

run("CUDA GEMM sanity", test_cuda_gemm)
run("CPU conv2d (fp32)", test_cpu_conv)
run("GPU conv2d with cuDNN (fp32)", test_gpu_cudnn_conv_fp32)
run("GPU conv2d WITHOUT cuDNN (fp32)", test_gpu_nocudnn_conv_fp32)
