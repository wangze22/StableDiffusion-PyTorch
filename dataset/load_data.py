import os
import json
from glob import glob
from pathlib import Path

# Allow PyTorch/Matplotlib to coexist with duplicate OpenMP runtimes on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import webdataset as wds
import matplotlib.pyplot as plt


def build_loader():
    root = Path(r"D:\datasets\huggingface\data_512_2M")
    shards = sorted(glob(str(root / "data_*.tar")))
    if not shards:
        raise FileNotFoundError(f"没有找到 {root} 下的 data_*.tar")

    urls = [f"file:{Path(p).as_posix()}" for p in shards]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])

    def parse_caption(j):
        if isinstance(j, dict):
            info = j
        elif isinstance(j, (bytes, bytearray)):
            info = json.loads(j.decode("utf-8"))
        elif isinstance(j, str):
            try:
                info = json.loads(j)
            except json.JSONDecodeError:
                info = {"caption": j}
        elif hasattr(j, "read"):
            info = json.loads(j.read().decode("utf-8"))
        else:
            info = {}

        for key in ["caption", "text", "prompt", "description", "prompt_text"]:
            if key in info and isinstance(info[key], str):
                return info[key]
        return ""

    def extract_subset(key):
        key_str = str(key)
        # 保留最后一段，避免包含路径前缀
        key_str = key_str.split("/")[-1]
        # 去掉可能的扩展名
        if "." in key_str:
            key_str = key_str.split(".", 1)[0]
        if "_" in key_str:
            base, suffix = key_str.rsplit("_", 1)
            if suffix.isdigit():
                return base
        return key_str

    dataset = (
        wds.WebDataset(urls, shardshuffle=False, handler=wds.ignore_and_continue)
        .shuffle(10000)
        .decode("pil")
        .to_tuple("jpg;png;jpeg", "json", "__key__")
        .map_tuple(transform, parse_caption, extract_subset)
    )

    loader = DataLoader(dataset, batch_size=8, num_workers=0, pin_memory=True)
    return loader


def show_images(images, captions):
    """把一个 batch 的图片和文字一起展示出来"""
    images = images[:8]  # 最多展示 8 张
    captions = captions[:8]
    plt.figure(figsize=(16, 8))
    for i, (img_tensor, caption) in enumerate(zip(images, captions)):
        plt.subplot(2, 4, i + 1)
        # 反归一化
        img = img_tensor * 0.5 + 0.5
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(caption[:50], fontsize=8, wrap=True)
    plt.tight_layout()
    plt.show()


def main():
    loader = build_loader()
    seen_subsets = set()
    for batch_idx, (images, captions, subsets) in enumerate(loader):
        batch_subsets = set(subsets)
        new_subsets = batch_subsets - seen_subsets
        if new_subsets:
            print("发现新的子数据集:", ", ".join(sorted(new_subsets)))
            seen_subsets.update(new_subsets)
        print(f"Batch {batch_idx} | {len(images)} samples | 子数据集: {', '.join(sorted(batch_subsets))}")
        print("Example caption:", captions[0])
        show_images(images, captions)
        if batch_idx == 2:
            break
    if seen_subsets:
        print("共发现的子数据集:", ", ".join(sorted(seen_subsets)))


if __name__ == "__main__":
    main()
