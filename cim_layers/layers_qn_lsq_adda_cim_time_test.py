# -*- coding: utf-8 -*-
"""
单独测试：ConvOnlyNet / LinearOnlyNet
使用 ModelProfiler(model, ...) 接口把已定义的 model 传入测试。
优先使用 line_profiler 做逐行热点；无则降级到 cProfile + torch.profiler。
"""
import io
import pstats
import cProfile
from pathlib import Path

import torch
import torch.nn as nn
from sympy import false
from torch.profiler import profile, ProfilerActivity
import sys, os, inspect
import time

# —— 正常 import —— #
import cim_layers.layers_qn_lsq_adda_cim as old
import cim_layers.layers_qn_lsq_adda_cim_opt as opt
from cim_layers.layers_utils_lsq import *
from cim_layers.layers_utils_adda import *
from cim_layers.quant_noise_utils import *
from cim_weight_mapper.weight_process import *

adc_bit = 8
dac_bit = 2
input_bit = 8
output_bit = 8
weight_bit = 4


# ========= 两个独立网络 =========
class ConvOnlyNet(nn.Module):
    def __init__(self, in_ch = 4, out_ch = 8, k = 3, device = "cuda"):
        super().__init__()
        self.conv = old.Conv2d_lsq_adda_cim(
            in_channels = in_ch, out_channels = out_ch,
            kernel_size = k, stride = 1, padding = k // 2, groups = 1,
            weight_bit = weight_bit, input_bit = input_bit, output_bit = output_bit,
            noise_scale = 0.0,
            dac_bit = dac_bit, adc_bit = adc_bit,
            adc_gain_1_scale = 1 / 63, adc_gain_range = [1, 255],
            adc_adjust_mode = 'gain',
            input_quant = True, output_quant = True, weight_quant = True,
            gain_noise_scale = 0.0, offset_noise_scale = 0.0, seed = 123, bias = True,
            ).to(device)

    def forward(self, x):
        return self.conv(x)


class LinearOnlyNet(nn.Module):
    def __init__(self, in_f = 128, out_f = 64, device = "cuda"):
        super().__init__()
        self.fc = old.Linear_lsq_adda_cim(
            in_features = in_f, out_features = out_f,
            weight_bit = weight_bit, input_bit = input_bit, output_bit = output_bit,
            noise_scale = 0.0,
            dac_bit = dac_bit, adc_bit = adc_bit,
            adc_gain_1_scale = 1 / 63, adc_gain_range = [1, 255],
            adc_adjust_mode = 'gain',
            input_quant = True, output_quant = True, weight_quant = True,
            gain_noise_scale = 0.0, offset_noise_scale = 0.0, seed = 321, bias = True,
            ).to(device)

    def forward(self, x):
        return self.fc(x)


# ========= 测试/分析类 =========
class ModelProfiler:
    """
    用法：
        mp = ModelProfiler(model, array_size=(128,128), device="cuda")
        mp.profile(inputs, tag="Conv forward")
    说明：
        - model: 你已经定义好的 nn.Module（ConvOnlyNet 或 LinearOnlyNet）
        - 会调用 map_weight_for_model(model, ...)
        - inputs: 前向需要的输入张量（Conv: [B,C,H,W]；Linear: [B,Fin]）
    """

    def __init__(self, model: nn.Module, array_size=(128, 128), device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 权重映射（按你的接口）
        self.model_weight_mapping_info = map_weight_for_model(
            self.model,
            draw_weight_block=False,
            array_size=array_size,
            weight_block_size=array_size,
            array_device_name='tester',
        )

    def _measure_wall_and_cuda_ms(self, runner, iters = 5):
        """
        用同一批 forward 同时测 wall-clock 与 CUDA events。
        - 先做 1 次预热（含同步）
        - 再连续跑 iters 次，在每次里同时记录 wall & cuda，最后取均值
        返回: (avg_wall_ms, avg_cuda_ms or None)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 预热一次，避免首次 kernel 初始化影响
        runner()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_sum = 0.0
        cuda_sum = 0.0
        have_cuda = torch.cuda.is_available()

        for _ in range(max(1, iters)):
            # WALL
            if have_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # CUDA events
            if have_cuda:
                stream = torch.cuda.current_stream()
                start_evt = torch.cuda.Event(enable_timing = True)
                end_evt = torch.cuda.Event(enable_timing = True)
                start_evt.record(stream)

            runner()

            if have_cuda:
                end_evt.record(stream)
                end_evt.synchronize()  # 等待该次 forward 的事件完成
                cuda_ms = start_evt.elapsed_time(end_evt)
            else:
                cuda_ms = None

            if have_cuda:
                torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000.0

            wall_sum += wall_ms
            if cuda_ms is not None:
                cuda_sum += cuda_ms

        avg_wall = wall_sum / max(1, iters)
        avg_cuda = (cuda_sum / max(1, iters)) if have_cuda else None
        return avg_wall, avg_cuda

    # ---------- 基础计时工具 ----------
    def _measure_total_ms(self, runner):
        """测一个 runner 的端到端耗时（含 CUDA 同步，保证准确）。"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        runner()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def _measure_cuda_ms(self, runner):
        """
        用 CUDA Events 测 GPU 时间（只在有 CUDA 的情况下有效）。
        注意：这测的是 GPU 时间线上的耗时，不包含 CPU 等待与隐式同步。
        """
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        # 记录事件到与 runner 相同的当前流
        start_evt.record(stream)
        runner()
        end_evt.record(stream)

        # 保证事件完成
        end_evt.synchronize()
        # 返回毫秒
        return start_evt.elapsed_time(end_evt)

    def _auto_collect_python_funcs(
        self,
        package_prefixes=("cim_layers", "cim_weight_mapper"),
        include_model_methods=True,
        max_funcs=1000,
    ):
        """
        返回一组可被 line_profiler 注册的 Python 函数/方法对象。
        规则：
          - 仅收集“纯 Python”函数/方法（有 __code__）
          - 模块名以 package_prefixes 开头，或源码路径在当前工程目录内
          - 可选：加入当前 model 的所有可分析方法
        """
        results = []
        seen = set()

        # 工程根目录：脚本所在目录
        try:
            proj_root = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            proj_root = os.getcwd()

        def _want_obj(obj):
            if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
                return False
            if not hasattr(obj, "__code__"):
                return False
            modname = getattr(obj, "__module__", "") or ""
            if any(modname.startswith(p) for p in package_prefixes):
                return True
            filename = inspect.getsourcefile(obj) or inspect.getfile(obj)
            if filename:
                try:
                    filename = os.path.abspath(filename)
                    if os.path.commonpath([proj_root, filename]) == proj_root:
                        return True
                except Exception:
                    pass
            return False

        def _add_obj(obj):
            try:
                key = obj.__code__
                if key in seen:
                    return
                seen.add(key)
                results.append(obj)
            except Exception:
                pass

        # 1) 扫描已导入模块
        for mod in list(sys.modules.values()):
            if not mod or not hasattr(mod, "__name__"):
                continue
            mname = mod.__name__
            if not any(mname.startswith(p) for p in package_prefixes):
                continue
            for _, func in inspect.getmembers(mod, inspect.isfunction):
                if _want_obj(func):
                    _add_obj(func)
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if getattr(cls, "__module__", "") != mname:
                    continue
                for _, f in cls.__dict__.items():
                    cand = None
                    if inspect.isfunction(f):
                        cand = f
                    elif isinstance(f, staticmethod) and inspect.isfunction(f.__func__):
                        cand = f.__func__
                    elif isinstance(f, classmethod) and inspect.isfunction(f.__func__):
                        cand = f.__func__
                    if cand and _want_obj(cand):
                        _add_obj(cand)

        # 2) 加入当前 model 的可分析方法
        if include_model_methods and hasattr(self, "model") and self.model is not None:
            for attr in dir(self.model):
                try:
                    fn = getattr(self.model, attr)
                except Exception:
                    continue
                if (inspect.ismethod(fn) or inspect.isfunction(fn)) and hasattr(fn, "__code__"):
                    if _want_obj(fn):
                        _add_obj(fn)

            cls = self.model.__class__
            for _, f in cls.__dict__.items():
                cand = None
                if inspect.isfunction(f):
                    cand = f
                elif isinstance(f, staticmethod) and inspect.isfunction(f.__func__):
                    cand = f.__func__
                elif isinstance(f, classmethod) and inspect.isfunction(f.__func__):
                    cand = f.__func__
                if cand and _want_obj(cand):
                    _add_obj(cand)

        if len(results) > max_funcs:
            results = results[:max_funcs]
        return results

    # ---------- 行级分析（若已安装 line_profiler） ----------
    def _try_line_profile(self, runner, extra_funcs=None, topk=20, title=""):
        try:
            import line_profiler
        except Exception:
            return False

        lp = line_profiler.LineProfiler()

        def _maybe_add(func):
            try:
                if func is not None and hasattr(func, "__code__"):
                    lp.add_function(func)
            except Exception:
                pass

        _maybe_add(self.model.forward)

        auto_funcs = self._auto_collect_python_funcs(
            package_prefixes=("cim_layers", "cim_weight_mapper"),
            include_model_methods=True,
            max_funcs=1000,
        )
        for f in auto_funcs:
            _maybe_add(f)

        if extra_funcs:
            for f in extra_funcs:
                _maybe_add(f)

        wrapped = lp(runner)
        wrapped()

        # —— 打印逐函数逐行统计 —— #
        s = io.StringIO()
        print(f"\n==== Line Profiler（逐行）: {title} ====")
        lp.print_stats(stream=s, stripzeros=True)
        print(s.getvalue())

        # —— 聚合全局最热行 —— #
        stats = lp.get_stats()
        timings = stats.timings
        unit = getattr(stats, "unit", 1.0)

        global_rows = []  # (file, func, lineno, t_total_sec, nhits, t_per_hit_sec)

        def _normalize_key(key):
            if hasattr(key, "co_filename"):
                return key.co_filename, key.co_name
            if isinstance(key, tuple) and len(key) >= 3:
                filename, _start_line, funcname = key[:3]
                return filename, funcname
            return str(key), "<unknown>"

        def _iter_line_records(timing_val):
            if isinstance(timing_val, dict):
                for lineno, triple in timing_val.items():
                    if isinstance(triple, (list, tuple)):
                        if len(triple) >= 3:
                            nhits, t_total, t_per_hit = triple[0], triple[1], triple[2]
                        else:
                            nhits, t_total = triple[0], triple[1]
                            t_per_hit = (t_total / nhits) if nhits else 0.0
                    else:
                        continue
                    yield lineno, nhits, float(t_total) * unit, float(t_per_hit) * unit
            else:
                for rec in timing_val:
                    if not isinstance(rec, (list, tuple)) or len(rec) < 3:
                        continue
                    lineno = rec[0]
                    nhits = rec[1]
                    t_total = rec[2]
                    if len(rec) >= 4:
                        t_per_hit = rec[3]
                    else:
                        t_per_hit = (t_total / nhits) if nhits else 0.0
                    yield lineno, nhits, float(t_total) * unit, float(t_per_hit) * unit

        for key, timing_val in timings.items():
            filename, funcname = _normalize_key(key)
            for lineno, nhits, t_total_sec, t_per_hit_sec in _iter_line_records(timing_val):
                global_rows.append((filename, funcname, lineno, t_total_sec, nhits, t_per_hit_sec))

        if not global_rows:
            return True

        global_rows.sort(key=lambda r: r[3], reverse=True)
        total_all = sum(r[3] for r in global_rows) or 1.0
        # total_wall_ms = self._measure_total_ms(runner)            # 端到端总时间
        # total_cuda_ms = self._measure_cuda_ms(runner)             # ★ GPU 真实耗时（可能为 None）
        total_wall_ms, total_cuda_ms = self._measure_wall_and_cuda_ms(runner, iters = 5)

        print(f"==== Line Profiler（全局最热行，按耗时降序，Top {topk}）: {title} ====")
        print(f"Total elapsed (one run): {total_wall_ms:.3f} ms")
        if total_cuda_ms is not None:
            print(f"GPU elapsed (CUDA events): {total_cuda_ms:.3f} ms")
        print(f"{'File:Line (Func)':100} {'Time(ms)':>10} {'%':>7} {'Hits':>7} {'perHit(us)':>12}")
        print("-" * 100)
        for file, func, lineno, t_total_sec, nhits, t_per_hit_sec in global_rows[:topk]:
            ms = t_total_sec * 1000.0
            pct = (t_total_sec / total_all) * 100.0
            us_per = t_per_hit_sec * 1e6
            print(
                f"{Path(file).name}:{lineno} ({func})".ljust(100),
                f"{ms:10.3f} {pct:7.2f} {nhits:7d} {us_per:12.1f}"
            )

        return True

    # ---------- 函数级 + 算子级（降级路径） ----------
    def _cprofile_and_opprofile(self, runner, title=""):
        # total_wall_ms = self._measure_total_ms(runner)
        # total_cuda_ms = self._measure_cuda_ms(runner)
        total_wall_ms, total_cuda_ms = self._measure_wall_and_cuda_ms(runner, iters = 5)

        print(f"\n[Total elapsed (one run)]: {total_wall_ms:.3f} ms -- {title}")
        if total_cuda_ms is not None:
            print(f"[GPU elapsed (CUDA events)]: {total_cuda_ms:.3f} ms -- {title}")

        # cProfile（函数级）
        pr = cProfile.Profile()
        pr.enable()
        runner()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(25)
        print(f"\n==== cProfile: {title} (Top 25 by cumulative time) ====")
        print(s.getvalue())

        # torch.profiler（算子级, CPU）
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False, with_stack=False) as prof:
            runner()
        events = prof.key_averages()
        total = sum(e.self_cpu_time_total for e in events) or 1.0
        rows = sorted(
            [(e.key, e.count, e.self_cpu_time_total / 1000.0, (e.self_cpu_time_total / total * 100.0))
             for e in events],
            key=lambda r: r[2], reverse=True,
        )
        print(f"==== Torch Profiler (CPU ops): {title} ====")
        print(f"{'Op':60} {'Calls':>7} {'Self CPU ms':>12} {'%':>6}")
        print("-" * 92)
        for name, calls, ms, pct in rows[:20]:
            print(f"{name:60} {calls:7d} {ms:12.3f} {pct:6.2f}")

    # ---------- 对外方法 ----------
    def profile(self, inputs, title="model forward", warmup=True, topk=20):
        """
        inputs: 前向所需的张量或元组
        """
        self.model.eval()

        def runner():
            with torch.no_grad():
                if isinstance(inputs, (tuple, list)):
                    _ = self.model(*inputs)
                else:
                    _ = self.model(inputs)

        if warmup:
            runner()

        extra_funcs = [
            floor_pass, floor_no_pass, clamp_pass, round_pass,
            init_adc_gain, add_adc_noise, input_quant, weight_quant_noise, output_quant, bit_split_tensor,
        ]
        self._try_line_profile(runner, extra_funcs=extra_funcs, topk=topk, title=title)
        self._cprofile_and_opprofile(runner, title=title)


# ========= 示例：分别测试卷积网络与全连接网络 =========
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 卷积网络（单层卷积）
    conv_net = ConvOnlyNet(in_ch = 128, out_ch = 128, k = 3, device = device)

    conv_net.conv.step_size_input.data = torch.tensor(0.04345978796482086)
    conv_net.conv.step_size_weight.data = torch.tensor(0.004208957310765982)
    conv_net.conv.adc_gain.data = torch.tensor(21.0)
    conv_net.conv.step_size_output.data = torch.tensor(0.025827452540397644)
    conv_net.conv._adc_gain_inited = True

    conv_prof = ModelProfiler(conv_net, array_size = (576, 128), device = device)
    x_conv = torch.randn(64, 128, 64, 64, device = device)
    conv_prof.profile(x_conv, title = "ConvOnlyNet forward", warmup = True, topk = 20)

    # 2) 全连接网络（单层全连接）
    # linear_net = LinearOnlyNet(in_f = 128, out_f = 64, device = device)
    # linear_prof = ModelProfiler(linear_net, array_size = (128, 128), device = device)
    # x_lin = torch.randn(2, 128, device = device)
    # linear_prof.profile(x_lin, title = "LinearOnlyNet forward", warmup = True, topk = 20)
