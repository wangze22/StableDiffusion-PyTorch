from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME
import torch, os

use_cuda = (CUDA_HOME is not None) and torch.cuda.is_available()

sources = ["bitsplit_dispatch.cpp", "bitsplit_cpu.cpp"]
ext_cls = CppExtension
# 精简 C++ 编译参数：去掉 -fopenmp（MSVC 不认），避免无关干扰
extra_compile_args = {"cxx": ["/O2"]}  # 或者啥都不写也行

if use_cuda:
    sources.append("bitsplit_cuda.cu")
    ext_cls = CUDAExtension
    # 关键：仅生成 PTX （compute_120），运行时由驱动 JIT
    extra_compile_args["nvcc"] = [
        "-O3",
        "-gencode=arch=compute_120,code=compute_120",  # 只生成 PTX，由驱动 JIT
        "--keep",  # 保留 .ptx / .cudafe1.cpp 等中间文件
        "-Xptxas", "-v",  # 打印 PTXAS 详细信息（寄存器/错误）
        ]

setup(
    name="bitsplit_ext",
    ext_modules=[
        ext_cls(
            "bitsplit_ext",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
