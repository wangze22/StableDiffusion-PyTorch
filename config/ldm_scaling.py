import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class LDMScalingBase:
    down_channels: Sequence[int] = (256, 384, 512, 768)
    mid_channels: Sequence[int] = (768, 512)
    time_emb_dim: int = 512
    conv_out_channels: int = 128
    num_heads: int = 16
    norm_channels: int = 32


def _round_to_multiple(value: float, multiple: int, *, min_value: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if min_value <= 0:
        raise ValueError("min_value must be positive")
    scaled = max(value, float(min_value))
    rounded = int(math.floor((scaled + multiple / 2.0) / multiple)) * multiple
    return max(multiple, rounded)


def _pick_divisor(candidates: Iterable[int], values: Sequence[int], *, name: str) -> int:
    sorted_candidates = sorted(set(int(c) for c in candidates if c > 0), reverse=True)
    for candidate in sorted_candidates:
        if candidate > min(values):
            continue
        if all(val % candidate == 0 for val in values):
            return candidate
    raise ValueError(f"Unable to pick {name}; tried {sorted_candidates} for values {values}")


def build_ldm_scaling(
        c_factor: float,
        *,
        channel_align: int = 16,
        min_channel: int = 32,
        time_align: int = 16,
        base: LDMScalingBase = LDMScalingBase(),
        head_candidates: Sequence[int] = (16, 12, 8, 6, 4, 2, 1),
        group_candidates: Sequence[int] = (32, 24, 16, 12, 8, 6, 4, 2, 1),
) -> Dict[str, object]:
    if c_factor <= 0:
        raise ValueError("c_factor must be > 0")
    if channel_align <= 0 or time_align <= 0:
        raise ValueError("Align values must be positive")

    def scale_channels(values: Sequence[int]) -> List[int]:
        scaled: List[int] = []
        for value in values:
            target = value / c_factor
            rounded = _round_to_multiple(target, channel_align, min_value=min_channel)
            scaled.append(int(rounded))
        return scaled

    down_channels = scale_channels(base.down_channels)
    mid_channels = [down_channels[-1], down_channels[-2]]

    conv_out_channels = _round_to_multiple(
        base.conv_out_channels / c_factor,
        channel_align,
        min_value=min_channel // 2 if min_channel > channel_align else channel_align,
    )

    time_emb_dim = _round_to_multiple(base.time_emb_dim / c_factor, time_align, min_value=time_align)
    if time_emb_dim % 2 != 0:
        time_emb_dim += time_align

    channels_for_heads: Tuple[int, ...] = tuple(down_channels + list(mid_channels))
    num_heads = _pick_divisor(head_candidates, channels_for_heads, name="num_heads")

    channels_for_groups = tuple(down_channels + list(mid_channels) + [conv_out_channels])
    norm_channels = _pick_divisor(group_candidates, channels_for_groups, name="norm_channels")

    return {
        "down_channels": down_channels,
        "mid_channels": mid_channels,
        "time_emb_dim": time_emb_dim,
        "conv_out_channels": conv_out_channels,
        "num_heads": num_heads,
        "norm_channels": norm_channels,
    }


def _format_dict(config: Dict[str, object]) -> str:
    return "\n".join(f"{key}: {config[key]}" for key in sorted(config))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LDM config safely for a given c_factor.")
    parser.add_argument("--c-factor", type=float, required=True, help="Compression factor to apply.")
    parser.add_argument("--channel-align", type=int, default=16, help="Round channels to this multiple.")
    parser.add_argument("--time-align", type=int, default=16, help="Round time embedding to this multiple.")
    parser.add_argument("--min-channel", type=int, default=32, help="Minimum allowed channel count.")
    args = parser.parse_args()

    config = build_ldm_scaling(
        c_factor=args.c_factor,
        channel_align=args.channel_align,
        time_align=args.time_align,
        min_channel=args.min_channel,
    )
    print(_format_dict(config))


if __name__ == "__main__":
    main()
