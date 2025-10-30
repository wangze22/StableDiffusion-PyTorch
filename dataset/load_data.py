import os
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

# Allow PyTorch/Matplotlib to coexist with duplicate OpenMP runtimes on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import webdataset as wds
import matplotlib.pyplot as plt


DATA_ROOT = Path(r"D:\datasets\huggingface\data_512_2M")
_summary_limit_env = os.getenv("DATASET_SUMMARY_LIMIT")
SUMMARY_SAMPLE_LIMIT = int(_summary_limit_env) if _summary_limit_env else None
if SUMMARY_SAMPLE_LIMIT is not None and SUMMARY_SAMPLE_LIMIT <= 0:
    SUMMARY_SAMPLE_LIMIT = None


def extract_subset(key):
    """Derive subset name from a WebDataset key such as flux_512_100k_00000000."""
    key_str = str(key).split("/")[-1]
    if "." in key_str:
        key_str = key_str.split(".", 1)[0]
    if "_" in key_str:
        base, suffix = key_str.rsplit("_", 1)
        if suffix.isdigit():
            return base
    return key_str


def get_shards(root: Path = DATA_ROOT):
    shards = sorted(root.glob("data_*.tar"))
    if not shards:
        raise FileNotFoundError(f"没有找到 {root} 下的 data_*.tar")
    return shards


def summarize_dataset(shards, sample_limit=None):
    subset_counts = Counter()
    subset_formats = defaultdict(Counter)
    total_files = 0
    processed_samples = 0

    limit_reached = False
    for shard_path in shards:
        with tarfile.open(shard_path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.split("/")[-1]
                if "." not in name:
                    continue
                stem, ext = name.rsplit(".", 1)
                ext = ext.lower()
                subset = extract_subset(stem)
                subset_formats[subset][ext] += 1
                total_files += 1
                if ext == "json":
                    subset_counts[subset] += 1
                    processed_samples += 1
                    if sample_limit and processed_samples >= sample_limit:
                        limit_reached = True
                        break
            if limit_reached:
                break

    return {
        "subset_counts": subset_counts,
        "subset_formats": subset_formats,
        "total_files": total_files,
        "total_samples": sum(subset_counts.values()),
        "sample_limit_hit": limit_reached,
    }


def build_loader(shards=None):
    if shards is None:
        shards = get_shards()

    urls = [f"file:{p.as_posix()}" for p in shards]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
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
    shards = get_shards()
    summary = summarize_dataset(shards, sample_limit=SUMMARY_SAMPLE_LIMIT)

    subset_counts = summary["subset_counts"]
    subset_formats = summary["subset_formats"]
    total_samples = summary["total_samples"]

    print(f"数据目录: {DATA_ROOT}")
    print(f"总样本数: {total_samples} | 子数据集数量: {len(subset_counts)}")
    for subset, count in subset_counts.most_common():
        formats = subset_formats[subset]
        format_desc = ", ".join(f"{ext}:{formats[ext]}" for ext in sorted(formats))
        print(f"  - {subset}: {count} samples ({format_desc})")
    if summary["sample_limit_hit"]:
        print("[警告] 统计只遍历到 sample_limit，数据量为估计值。")

    loader = build_loader(shards=shards)
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
