from pathlib import Path

from PIL import Image


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def resize_images(input_dir: Path, output_dir: Path, size: tuple[int, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = size

    image_files = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    total = len(image_files)
    if total == 0:
        print("没有找到可用的图片文件")
        return

    print(f"共找到 {total} 张图片，开始处理...")

    for idx, src in enumerate(image_files, start=1):
        rel_path = src.relative_to(input_dir)
        dst_path = (output_dir / rel_path).with_suffix(".jpg")
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                resized = img.resize((width, height), Image.LANCZOS)
                resized.save(dst_path, format="JPEG", quality=95)
            if idx % 10 == 0 or idx == total:
                print(f"[{idx}/{total}] 完成 {src}")
        except Exception as exc:
            print(f"跳过 {src}: {exc}")


def main() -> None:
    # TODO: 修改下面这三个变量即可
    src_dir = r"D:\datasets\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img"
    dst_dir = r"D:\datasets\CelebAMask-HQ\CelebAMask-HQ\256"
    target_size = [256, 256]

    if len(target_size) != 2:
        raise ValueError("target_size 需要形如 [width, height]")

    input_dir = Path(src_dir).expanduser().resolve()
    output_dir = Path(dst_dir).expanduser().resolve()
    size = int(target_size[0]), int(target_size[1])

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    resize_images(input_dir, output_dir, size)


if __name__ == "__main__":
    main()
