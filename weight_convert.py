import os
import re
import torch
from collections import OrderedDict, defaultdict

# -----------------------------
# 权重转换核心逻辑（保持原样）
# -----------------------------
PATTERN = re.compile(r"^(?P<prefix>.*?)(?:group_linears)\.(?P<idx>\d+)\.(?P<what>weight|bias)$")

def _stack_group_params(items, *, key_prefix):
    if not items:
        return None
    idxs = sorted(items.keys())
    assert idxs == list(range(len(idxs))), f"indices not contiguous: {idxs}"
    tensors = [items[i] for i in idxs]
    shapes = {tuple(t.shape) for t in tensors}
    assert len(shapes) == 1, f"Shape mismatch under '{key_prefix}': {shapes}"
    return torch.stack(tensors, dim=0)

def convert_state_dict(old_sd: OrderedDict) -> OrderedDict:
    new_sd = OrderedDict(old_sd)
    buckets = defaultdict(lambda: {"w": {}, "b": {}})

    for k, v in list(old_sd.items()):
        m = PATTERN.match(k)
        if not m:
            continue
        prefix = m.group("prefix")
        idx = int(m.group("idx"))
        what = m.group("what")
        if what == "weight":
            buckets[prefix]["w"][idx] = v
        else:
            buckets[prefix]["b"][idx] = v

    for prefix, wb in buckets.items():
        W = _stack_group_params(wb["w"], key_prefix=prefix + "weight")
        B = _stack_group_params(wb["b"], key_prefix=prefix + "bias")
        assert W is not None and B is not None

        new_weight_key = prefix + "weight"
        new_bias_key = prefix + "bias"

        new_sd[new_weight_key] = W.contiguous()
        new_sd[new_bias_key] = B.contiguous()

        for i in range(W.shape[0]):
            wk = f"{prefix}group_linears.{i}.weight"
            bk = f"{prefix}group_linears.{i}.bias"
            new_sd.pop(wk, None)
            new_sd.pop(bk, None)

    return new_sd

def load_any(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj, obj["state_dict"], True
    elif isinstance(obj, (dict, OrderedDict)):
        return None, obj, False
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

def save_any(path, full_ckpt, new_sd, was_wrapped):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if was_wrapped:
        full_ckpt["state_dict"] = new_sd
        torch.save(full_ckpt, path)
    else:
        torch.save(new_sd, path)

def default_out_path(in_path: str) -> str:
    base, ext = os.path.splitext(in_path)
    if not ext:
        ext = ".pt"
    return f"{base}._glfast{ext}"

def convert_file(in_path: str, out_path: str = None):
    print(f"[INFO] Loading: {in_path}")
    full_ckpt, sd, was_wrapped = load_any(in_path)

    print("[INFO] Converting grouped linear weights ...")
    new_sd = convert_state_dict(OrderedDict(sd))

    if out_path is None:
        out_path = default_out_path(in_path)

    print(f"[INFO] Saving to: {out_path}")
    save_any(out_path, full_ckpt if was_wrapped else None, new_sd, was_wrapped)
    print("[DONE] Conversion finished successfully!")

# -----------------------------
# 主程序（不再使用 argparse）
# -----------------------------
def main():
    # 在这里直接设置输入路径
    input_path = 'runs_tc05_qn_train_server/ddpm_20251026-062209/LSQ_AnDi/0.0800/ddpm_ckpt_text_image_cond_clip.pth'
    output_path = None  # 默认加后缀 _glfast.pt，可改成具体路径字符串

    convert_file(input_path, output_path)

if __name__ == "__main__":
    main()
