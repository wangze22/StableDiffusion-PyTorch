from collections import OrderedDict

import torch

def convert_container(obj, converted_modules):
    if isinstance(obj, (dict, OrderedDict)):
        return convert_mapping(obj, converted_modules)

    if isinstance(obj, list):
        return [convert_container(item, converted_modules) for item in obj]

    if isinstance(obj, tuple):
        return tuple(convert_container(item, converted_modules) for item in obj)

    return obj


def convert_mapping(mapping, converted_modules):
    new_mapping = mapping.__class__()
    for key, value in mapping.items():
        if isinstance(key, str):
            if key.endswith(".in_proj_weight"):
                base = key[: -len(".in_proj_weight")]
                weight = value
                if weight.size(0) % 3 != 0:
                    raise ValueError(f"Unexpected shape for {key}: {tuple(weight.shape)}")
                q_w, k_w, v_w = weight.chunk(3, dim=0)
                new_mapping[f"{base}.q_proj.weight"] = q_w.contiguous()
                new_mapping[f"{base}.k_proj.weight"] = k_w.contiguous()
                new_mapping[f"{base}.v_proj.weight"] = v_w.contiguous()
                converted_modules.add(base)
                continue

            if key.endswith(".in_proj_bias"):
                base = key[: -len(".in_proj_bias")]
                bias = value
                if bias.size(0) % 3 != 0:
                    raise ValueError(f"Unexpected shape for {key}: {tuple(bias.shape)}")
                q_b, k_b, v_b = bias.chunk(3, dim=0)
                new_mapping[f"{base}.q_proj.bias"] = q_b.contiguous()
                new_mapping[f"{base}.k_proj.bias"] = k_b.contiguous()
                new_mapping[f"{base}.v_proj.bias"] = v_b.contiguous()
                converted_modules.add(base)
                continue

        new_mapping[key] = convert_container(value, converted_modules)

    return new_mapping


def main():
    input_path = INPUT_PATH.strip()
    output_path = OUTPUT_PATH.strip()

    if not input_path:
        raise ValueError("Please set INPUT_PATH to the original checkpoint path.")
    if not output_path and not DRY_RUN:
        raise ValueError("Please set OUTPUT_PATH to the destination checkpoint path.")

    checkpoint = torch.load(input_path, map_location="cpu")
    converted_modules = set()

    converted_checkpoint = convert_container(checkpoint, converted_modules)

    if not converted_modules:
        print("No attention layers required conversion. The checkpoint appears to be up to date.")
    else:
        print(f"Converted {len(converted_modules)} attention module(s):")
        for name in sorted(converted_modules):
            print(f"  - {name}")

    if not DRY_RUN:
        torch.save(converted_checkpoint, output_path)
        print(f"Converted checkpoint saved to {output_path}")
    else:
        print("Dry run complete. No file written.")


if __name__ == "__main__":
    # Edit these paths before running the script.
    # Example:
    # INPUT_PATH = r"C:\path\to\old_weights.pt"
    # OUTPUT_PATH = r"C:\path\to\converted_weights.pt"
    INPUT_PATH = 'runs_tc05_qn_train_server/ddpm_20251028-195206_save_condition_0.5/LSQ_AnDi/0.0800/ddpm_ckpt_text_image_cond_clip.pth'
    OUTPUT_PATH ='runs_tc05_qn_train_server/ddpm_20251028-195206_save_condition_0.5/LSQ_AnDi/0.0800/ddpm_ckpt_text_image_cond_clip_qkv.pth'
    DRY_RUN = False

    main()
