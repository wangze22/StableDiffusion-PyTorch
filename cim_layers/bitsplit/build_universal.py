#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal one-click builder for PyTorch C++/CUDA extension.
- Windows/Linux 自动适配
- 自动导入 MSVC 环境（vcvars64.bat）
- CUDA 可用则启用，否则回退 CPU-only
- 默认安静输出（不刷屏），--verbose 可开启详细日志
Place this file with: bitsplit_cpu.cpp, bitsplit_dispatch.cpp, bitsplit_cuda.cu (可选)
"""

import os
import sys
import json
import shutil
import argparse
import platform
import subprocess
from pathlib import Path

# ---------------- basic logging ----------------
def info(*a): print("[build]", *a)
def warn(*a): print("[build][WARN]", *a)
def err (*a): print("[build][ERROR]", *a)

def which(p):
    from shutil import which as _which
    return _which(p)

# ---------------- env helpers ----------------
def get_cuda_home():
    for k in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(k)
        if v and Path(v).exists():
            return v
    for p in ("/usr/local/cuda", "/opt/cuda"):
        if Path(p).exists():
            return p
    return None

def detect_arch_list(torch):
    v = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if v:
        return v
    archs = []
    try:
        if torch.cuda.is_available():
            maj, min_ = torch.cuda.get_device_capability()
            archs.append(f"{maj}.{min_}")
    except Exception:
        pass
    if not archs:
        archs = ["8.6", "8.9"]  # safe defaults; override via env if needed
    return ";".join(archs)

# --------- Windows: import MSVC env (vcvars) ----------
def _run_cmd_out(cmd):
    return subprocess.check_output(cmd, shell=True, text=True, encoding="utf-8", errors="ignore").strip()

def find_vs_vcvars():
    vswhere = which("vswhere") or r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if Path(vswhere).exists():
        try:
            j = _run_cmd_out(
                f'"{vswhere}" -latest -products * '
                f'-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json'
            )
            arr = json.loads(j or "[]")
            if arr:
                inst = arr[0]["installationPath"]
                for p in (
                    Path(inst) / "VC/Auxiliary/Build/vcvars64.bat",
                    Path(inst) / "VC/Auxiliary/Build/vcvarsall.bat",
                ):
                    if p.exists():
                        return str(p)
        except Exception:
            pass
    # fallbacks
    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def import_msvc_env_quiet():
    """Load MSVC env into current process on Windows; returns True if cl is available."""
    if platform.system() != "Windows":
        return True
    if which("cl"):
        return True
    bat = find_vs_vcvars()
    if not bat:
        warn("未找到 VS Build Tools。请安装 VS2022 Build Tools（含“使用 C++ 的桌面开发”与 Windows SDK）。")
        return False
    info("Loading MSVC environment via:", bat)
    # Call vcvars and capture environment, then inject into os.environ
    cmd = f'cmd /s /c "call \"{bat}\" >nul && set"'
    try:
        out = _run_cmd_out(cmd)
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v
    except subprocess.CalledProcessError:
        warn("调用 vcvars 失败；建议从“x64 Native Tools for VS 2022”命令行运行本脚本。")
        return False
    if which("cl"):
        info("MSVC environment imported.")
        return True
    warn("vcvars 已调用但仍找不到 cl。请从“x64 Native Tools for VS 2022”运行本脚本。")
    return False

# ---------------- build helpers ----------------
def copy_built_module(mod):
    so_path = Path(mod.__file__)
    dst = Path.cwd() / so_path.name
    if str(dst) != str(so_path):
        shutil.copy2(so_path, dst)
        info("Copied built artifact to:", dst)
    return dst

def try_build(sources, name, extra_cflags, extra_cuda_cflags, verbose, with_cuda):
    from torch.utils.cpp_extension import load
    mod = load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags if with_cuda else [],
        with_cuda=with_cuda,
        verbose=verbose,   # 默认 False（安静），--verbose 可打开
    )
    return copy_built_module(mod)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU-only build (no NVCC).")
    ap.add_argument("--clean", action="store_true", help="Clean caches before build.")
    ap.add_argument("--name", default="bitsplit_ext", help="Extension module name.")
    ap.add_argument("--no-fast-math", action="store_true", help="Disable fast-math.")
    ap.add_argument("--no-openmp", action="store_true", help="Disable OpenMP even if available.")
    ap.add_argument("--verbose", action="store_true", help="Verbose compile output.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    # 安静关键设置：让 Ninja 能隐藏 include 列表（Windows）
    if platform.system() == "Windows":
        os.environ.setdefault("VSLANG", "1033")  # 英文输出，Ninja 会吞掉 "Note: including file:"
    os.environ.setdefault("USE_NINJA", "1")      # 也可 set USE_NINJA=0 改用 distutils（更安静，但可能更慢）

    # 源码检测
    src_dir = Path.cwd()
    base_srcs = [src_dir / "bitsplit_cpu.cpp", src_dir / "bitsplit_dispatch.cpp"]
    missing = [str(p) for p in base_srcs if not p.exists()]
    if missing:
        err("缺少基础源码文件:", ", ".join(missing))
        sys.exit(1)
    srcs = base_srcs.copy()
    cu_src = src_dir / "bitsplit_cuda.cu"
    if args.cpu_only:
        info("Forcing CPU-only build.")
    else:
        if not cu_src.exists():
            err("bitsplit_cuda.cu 不存在，无法进行 CUDA 构建。")
            sys.exit(1)
        srcs.append(cu_src)

    # 清理
    if args.clean:
        info("Cleaning build caches...")
        try:
            from torch.utils.cpp_extension import _get_build_directory
        except Exception:
            _get_build_directory = None
        shutil.rmtree("build", ignore_errors=True)
        if _get_build_directory:
            try:
                bdir = Path(_get_build_directory(args.name, verbose=False))
                shutil.rmtree(bdir, ignore_errors=True)
                info("Removed JIT cache:", bdir)
            except Exception:
                pass
        for pat in (f"{args.name}*.so", f"{args.name}*.pyd", f"{args.name}*.dll", f"{args.name}*.dylib"):
            for f in Path.cwd().glob(pat):
                try:
                    f.unlink()
                except Exception:
                    pass
        info("Clean done.")

    # PyTorch
    try:
        import torch
        info("PyTorch:", torch.__version__, "CUDA in torch:", torch.version.cuda)
    except Exception:
        err("需要已安装的 PyTorch。若需 CUDA 请安装带 CUDA 的发行版。")
        sys.exit(1)

    # Windows: 导入 MSVC 环境
    if platform.system() == "Windows" and not import_msvc_env_quiet():
        err("MSVC 环境未就绪，无法继续。")
        sys.exit(2)

    # CUDA 可用性
    cuda_home = get_cuda_home()
    nvcc = which("nvcc")
    if not args.cpu_only:
        if cuda_home is None or not bool(torch.version.cuda):
            err("未检测到完整的 CUDA 工具链或 PyTorch CUDA 运行时，无法进行 GPU 构建。")
            sys.exit(3)
        if nvcc is None:
            candidate = Path(cuda_home) / "bin" / ("nvcc.exe" if platform.system() == "Windows" else "nvcc")
            if candidate.exists():
                nvcc = str(candidate)
                os.environ["PATH"] = f"{candidate.parent}{os.pathsep}" + os.environ.get("PATH", "")
        if nvcc is None:
            err("未在 PATH 或 CUDA_HOME/bin 中找到 nvcc，可使用 --cpu-only 手动回退。")
            sys.exit(3)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", detect_arch_list(torch))
        info("CUDA_HOME:", cuda_home)
        info("NVCC:", nvcc)
        info("TORCH_CUDA_ARCH_LIST:", os.environ["TORCH_CUDA_ARCH_LIST"])
    with_cuda = not args.cpu_only

    # 编译参数（默认安静且高优化）
    extra_cflags, extra_cuda_cflags = [], []
    is_windows = platform.system() == "Windows"

    if is_windows:
        extra_cflags += ["/O2"]
        if not args.no_openmp:
            # 现代 OpenMP（老 MSVC 可自动回退在我们的多尝试里）
            extra_cflags += ["/openmp:llvm"]
        if not args.no_fast_math:
            extra_cflags += ["/fp:fast"]
    else:
        extra_cflags += ["-O3", "-march=native"]
        if not args.no_openmp:
            extra_cflags += ["-fopenmp"]
        if not args.no_fast_math:
            extra_cflags += ["-ffast-math"]

    if with_cuda:
        extra_cuda_cflags += ["-O3"]
        if not args.no_fast_math:
            extra_cuda_cflags += ["--use_fast_math"]
        # 刻意不加 "-Xptxas -v"（会大量噪声）

    # 构建尝试矩阵（含若干回退）
    sources = [str(p) for p in srcs]
    info("Sources:", sources)

    build_mode = "CUDA" if with_cuda else "CPU-only"
    try:
        info("Building:", build_mode)
        dst = try_build(
            sources=sources,
            name=args.name,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=with_cuda,
            verbose=args.verbose,
        )
        info("SUCCESS:", build_mode)
        info("Built:", dst)
    except Exception as e:
        err("Build failed:")
        print("\n==== Error detail ====\n")
        print(e)
        print("\n======================\n")
        sys.exit(2)

if __name__ == "__main__":
    main()
