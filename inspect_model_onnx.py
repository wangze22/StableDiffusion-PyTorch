import importlib
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node, symbolic_trace
try:
    from torch.fx.experimental.shape_prop import ShapeProp
except ImportError:
    try:
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError:
        ShapeProp = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RUNTIME_CONFIG: Dict[str, Any] = {
    "device": "cpu",
    "seed": 2025,
    "enable_torchinfo": True,
    "enable_module_tree": True,
    "enable_structure_graph": True,
    "enable_onnx": False,
    "graphviz_bin_dir": None,  # e.g. r"C:/Program Files/Graphviz/bin"
}

MODEL_SPEC: Dict[str, Any] = {
    # Fully qualified path to the model class
    "target": "models.blocks.DownBlock",
    # Keyword arguments used to instantiate the model
    "init_args": {
        "in_channels": 64,
        "out_channels": 128,
        "t_emb_dim": 256,
        "down_sample": True,
        "num_heads": 4,
        "num_layers": 3,
        "attn": True,
        "norm_channels": 32,
        "cross_attn": False,
        "context_dim": 128,
    },
    # Names of arguments passed positionally into forward(*args)
    "forward_args": ["x", "t_emb"],
    # Mapping of keyword arguments passed to forward(**kwargs). Values may reference
    # input names or be literal values.
    "forward_kwargs": {},
    "set_eval": True,
}

# Input specifications used both for dummy data creation and for labelling graph inputs.
# Each spec supports:
#   name        : str, matches a forward argument/kwarg name
#   kind        : 'tensor' (default) or 'scalar'
#   shape       : tuple/list of ints (for tensors)
#   dtype       : torch dtype name (default 'float32')
#   factory     : 'randn', 'zeros', or 'ones' (default 'randn')
#   value       : optional literal (overrides factory when provided)
#   requires_grad : bool (default False)
INPUT_SPECS: List[Dict[str, Any]] = [
    {"name": "x", "kind": "tensor", "shape": (1, 64, 32, 32), "dtype": "float32", "factory": "randn"},
    {"name": "t_emb", "kind": "tensor", "shape": (1, 256), "dtype": "float32", "factory": "randn"},
]

GRAPH_CONFIG: Dict[str, Any] = {
    "output_path": "visualizations/model_structure",
    "format": "pdf",
    "rankdir": "LR",
    "show_module_path": False,
    "show_shapes": True,
    "include_functions": {
        "operator.add",
        "torch.add",
        "aten.add.Tensor",
        "torch.cat",
    },
    "include_methods": set(),
    "skip_module_types": {"Identity"},
    "suppress_get_attr": True,
    "collapse_constant_nodes": True,
}

ONNX_CONFIG: Dict[str, Any] = {
    "output_path": "onnx_exports/model.onnx",
    "opset": 18,
    "input_names": None,  # default uses MODEL_SPEC['forward_args']
    "dynamic_axes": None,  # override as {input_name: {dim_index: "label"}}
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _select_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA device not available. Falling back to CPU.")
        device = torch.device("cpu")
    return device


def _import_symbol(target: str) -> Any:
    module_path, attr = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _ensure_package(pkg: str, install_hint: str) -> bool:
    if importlib.util.find_spec(pkg) is None:
        print(f"[Skip] Optional package '{pkg}' not found. Install via '{install_hint}' to enable this step.")
        return False
    return True


def _append_to_path(bin_dir: Path) -> None:
    path_str = os.environ.get("PATH", "")
    pieces = path_str.split(os.pathsep) if path_str else []
    if str(bin_dir) not in pieces:
        os.environ["PATH"] = os.pathsep.join(([str(bin_dir)] + pieces) if path_str else [str(bin_dir)])


def _ensure_graphviz_binaries(bin_dir: Optional[Union[str, Path]]) -> bool:
    if shutil.which("dot"):
        return True

    if bin_dir:
        candidate = Path(bin_dir)
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.is_dir():
            _append_to_path(candidate)
            if shutil.which("dot"):
                return True

    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\Graphviz\bin"),
            Path(r"C:\Program Files (x86)\Graphviz2.38\bin"),
            Path(r"C:\Program Files\Graphviz2.38\bin"),
        ]
    else:
        candidates = [Path("/usr/local/bin"), Path("/opt/homebrew/bin"), Path("/usr/bin")]

    for candidate in candidates:
        if candidate.is_dir():
            _append_to_path(candidate)
            if shutil.which("dot"):
                return True

    return False


def _parse_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError:
        raise ValueError(f"Unsupported dtype '{name}'")
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Attribute '{name}' is not a torch.dtype")
    return dtype


def _make_tensor(spec: Dict[str, Any], device: torch.device) -> torch.Tensor:
    dtype = _parse_dtype(spec.get("dtype", "float32"))
    shape = tuple(spec["shape"])
    factory = spec.get("factory", "randn")

    if "value" in spec:
        value = spec["value"]
        tensor = torch.tensor(value, dtype=dtype, device=device)
        if tensor.shape != torch.Size(shape):
            tensor = tensor.reshape(shape)
        return tensor

    if factory == "randn":
        tensor = torch.randn(*shape, device=device, dtype=dtype)
    elif factory == "zeros":
        tensor = torch.zeros(*shape, device=device, dtype=dtype)
    elif factory == "ones":
        tensor = torch.ones(*shape, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported tensor factory '{factory}'")

    if spec.get("requires_grad", False):
        tensor.requires_grad_(True)
    return tensor


def _build_inputs(
    specs: Sequence[Dict[str, Any]],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    values: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for spec in specs:
        name = spec["name"]
        kind = spec.get("kind", "tensor")
        if kind == "tensor":
            value = _make_tensor(spec, device)
        elif kind == "scalar":
            value = torch.tensor(spec.get("value", 0.0), dtype=_parse_dtype(spec.get("dtype", "float32")), device=device)
        else:
            raise ValueError(f"Unsupported input kind '{kind}' for '{name}'")
        values[name] = value
        meta[name] = spec
    return values, meta


def _prepare_call_arguments(
    model_spec: Dict[str, Any],
    input_values: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any], List[str], List[Any]]:
    forward_args_names: Sequence[str] = model_spec.get("forward_args", [])
    args: List[Any] = []
    for name in forward_args_names:
        if name not in input_values:
            raise KeyError(f"Input spec for positional argument '{name}' not found.")
        args.append(input_values[name])

    forward_kwargs_cfg = model_spec.get("forward_kwargs", {})
    kwargs: Dict[str, Any] = {}
    ordered_input_names: List[str] = list(forward_args_names)

    for key, value in forward_kwargs_cfg.items():
        if isinstance(value, str):
            if value not in input_values:
                raise KeyError(f"Input spec for keyword argument '{key}' (source '{value}') not found.")
            kwargs[key] = input_values[value]
            if value not in ordered_input_names:
                ordered_input_names.append(value)
        else:
            kwargs[key] = value

    ordered_input_values: List[Any] = [input_values[name] for name in ordered_input_names]
    return args, kwargs, ordered_input_names, ordered_input_values


def _gather_callable_name(node: Node) -> str:
    target = node.target
    if node.op == "call_function":
        if hasattr(target, "name"):
            # torch.ops OpOverload
            try:
                return target.name()
            except TypeError:
                pass
        module = getattr(target, "__module__", "")
        name = getattr(target, "__qualname__", getattr(target, "__name__", str(target)))
        return f"{module}.{name}" if module else name
    if node.op == "call_method":
        return str(target)
    if node.op == "call_module":
        return str(target)
    return node.op


def _format_shape(node: Node) -> str:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return ""

    def _shape_from_meta(m: Any) -> Optional[str]:
        shape = getattr(m, "shape", None)
        if shape is None:
            return None
        return "x".join(str(s) for s in shape)

    if isinstance(meta, (list, tuple)):
        shapes = [s for m in meta if (s := _shape_from_meta(m))]
        return "[" + ", ".join(shapes) + "]" if shapes else ""

    shape_str = _shape_from_meta(meta)
    return shape_str or ""


def _is_visible_node(
    node: Node,
    graph_config: Dict[str, Any],
    module_lookup: Dict[str, nn.Module],
) -> bool:
    if node.op == "output":
        return False
    if node.op == "placeholder":
        return True
    if node.op == "get_attr":
        return not graph_config.get("suppress_get_attr", True)
    if node.op == "call_module":
        module = module_lookup.get(str(node.target))
        if module is None:
            return True
        skip_types: Set[str] = set(graph_config.get("skip_module_types", []))
        return module.__class__.__name__ not in skip_types
    if node.op == "call_function":
        name = _gather_callable_name(node)
        include_functions: Set[str] = set(graph_config.get("include_functions", []))
        return name in include_functions
    if node.op == "call_method":
        include_methods: Set[str] = set(graph_config.get("include_methods", []))
        name = _gather_callable_name(node)
        return name in include_methods
    return False


def _visible_parents(
    node: Node,
    visibility: Dict[Node, bool],
) -> Set[Node]:
    result: Set[Node] = set()
    stack = list(node.all_input_nodes)
    visited: Set[Node] = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if visibility.get(current, False):
            result.add(current)
        else:
            stack.extend(current.all_input_nodes)
    return result


def _build_structure_graph(
    model: nn.Module,
    gm: GraphModule,
    input_meta: Dict[str, Dict[str, Any]],
    graph_config: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> None:
    if not runtime_config.get("enable_structure_graph", True):
        return
    if not _ensure_package("graphviz", "pip install graphviz"):
        return
    if not _ensure_graphviz_binaries(runtime_config.get("graphviz_bin_dir")):
        print(
            "\n[Warn] Graphviz executables not found. Add the Graphviz 'bin' directory to PATH or set "
            "RUNTIME_CONFIG['graphviz_bin_dir'] and rerun."
        )
        return

    from graphviz import Digraph

    module_lookup = dict(model.named_modules())
    visibility: Dict[Node, bool] = {}
    for node in gm.graph.nodes:
        visibility[node] = _is_visible_node(node, graph_config, module_lookup)

    output_path = Path(graph_config.get("output_path", "visualizations/model_structure"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dot = Digraph("ModelStructure", format=graph_config.get("format", "pdf"))
    dot.attr(rankdir=graph_config.get("rankdir", "LR"), fontsize="12", nodesep="0.6", ranksep="1.0")

    added_nodes: Set[str] = set()

    def add_node(node_id: str, label: str, shape: str, style: str, fillcolor: str) -> None:
        if node_id in added_nodes:
            return
        dot.node(node_id, label, shape=shape, style=style, fillcolor=fillcolor)
        added_nodes.add(node_id)

    for node in gm.graph.nodes:
        if not visibility.get(node, False):
            continue

        node_id = node.name
        label_lines: List[str] = []
        if node.op == "placeholder":
            spec = input_meta.get(node.name, {})
            label_lines.append(spec.get("name", node.name))
            if graph_config.get("show_shapes", True):
                shape_str = _format_shape(node)
                if not shape_str:
                    shape_spec = spec.get("shape")
                    if shape_spec:
                        shape_str = "x".join(str(s) for s in shape_spec)
                if shape_str:
                    label_lines.append(f"[{shape_str}]")
            add_node(node_id, "\n".join(label_lines), shape="box", style="filled", fillcolor="#e9f2fb")
            continue

        if node.op == "call_module":
            module: Optional[nn.Module] = module_lookup.get(str(node.target))
            class_name = module.__class__.__name__ if module else str(node.target)
            if graph_config.get("show_module_path", False):
                label_lines.append(str(node.target))
            label_lines.append(class_name)
            if graph_config.get("show_shapes", True):
                shape = _format_shape(node)
                if shape:
                    label_lines.append(f"[{shape}]")
            add_node(node_id, "\n".join(label_lines), shape="record", style="filled", fillcolor="#ffffff")
            continue

        if node.op in {"call_function", "call_method"}:
            label_lines.append(_gather_callable_name(node))
            if graph_config.get("show_shapes", True):
                shape = _format_shape(node)
                if shape:
                    label_lines.append(f"[{shape}]")
            add_node(node_id, "\n".join(label_lines), shape="oval", style="filled", fillcolor="#f7f7f7")
            continue

        if node.op == "get_attr":
            label_lines.append(str(node.target))
            add_node(node_id, "\n".join(label_lines), shape="box", style="filled", fillcolor="#fff4e5")

    output_node_id = "output"
    add_node(output_node_id, "Output", shape="box", style="filled", fillcolor="#e9f2fb")

    edges: Set[Tuple[str, str]] = set()
    for node in gm.graph.nodes:
        if node.op == "output":
            parents = _visible_parents(node, visibility)
            for parent in parents:
                if not visibility.get(parent, False):
                    continue
                edge = (parent.name, output_node_id)
                if edge not in edges:
                    dot.edge(*edge)
                    edges.add(edge)
            continue

        if not visibility.get(node, False):
            continue

        parents = _visible_parents(node, visibility)
        for parent in parents:
            if not visibility.get(parent, False):
                continue
            edge = (parent.name, node.name)
            if edge not in edges:
                dot.edge(*edge)
                edges.add(edge)

    rendered = dot.render(output_path.as_posix(), cleanup=True)
    print(f"\n[OK] Structure graph exported to {rendered}")


def _run_torchinfo(model: nn.Module, args: Sequence[Any], kwargs: Dict[str, Any]) -> None:
    if not _ensure_package("torchinfo", "pip install torchinfo"):
        return
    from torchinfo import summary

    print("\n--- TorchInfo Summary ---")
    summary_kwargs = dict(
        input_data=tuple(args),
        col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),
        depth=5,
        verbose=1,
    )
    if kwargs:
        summary_kwargs.update(kwargs)
    summary(model, **summary_kwargs)


def _print_module_tree(model: nn.Module) -> None:
    print("\n--- Module Tree ---")
    for name, module in model.named_modules():
        if name == "":
            continue
        extra = []
        if hasattr(module, "in_channels"):
            extra.append(f"in={getattr(module, 'in_channels')}")
        if hasattr(module, "out_channels"):
            extra.append(f"out={getattr(module, 'out_channels')}")
        if hasattr(module, "kernel_size"):
            extra.append(f"kernel={getattr(module, 'kernel_size')}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        print(f"{name}: {module.__class__.__name__}{suffix}")


def _export_to_onnx(
    model: nn.Module,
    input_names: List[str],
    input_values: Sequence[Any],
    forward_args_names: Sequence[str],
    forward_kwargs_cfg: Dict[str, Any],
    runtime_config: Dict[str, Any],
) -> None:
    if not runtime_config.get("enable_onnx", False):
        return
    if not _ensure_package("onnxscript", "pip install onnxscript"):
        return

    output_path = Path(ONNX_CONFIG["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configured_input_names = ONNX_CONFIG.get("input_names")
    effective_input_names = list(input_names if configured_input_names is None else configured_input_names)

    name_to_value = {name: value for name, value in zip(input_names, input_values)}
    export_args = tuple(name_to_value[name] for name in effective_input_names)

    dynamic_axes = ONNX_CONFIG.get("dynamic_axes")
    if dynamic_axes is None:
        dynamic_axes = {}
        for name in effective_input_names:
            spec = next((s for s in INPUT_SPECS if s.get("name") == name), None)
            if spec and spec.get("kind", "tensor") == "tensor" and len(spec.get("shape", [])) >= 1:
                dynamic_axes[name] = {0: "batch"}
        dynamic_axes["output"] = {0: "batch"}

    positional_indices = [effective_input_names.index(name) for name in forward_args_names]
    kwarg_indices = {
        key: effective_input_names.index(value)
        for key, value in forward_kwargs_cfg.items()
        if isinstance(value, str) and value in effective_input_names
    }
    kwarg_literals = {key: value for key, value in forward_kwargs_cfg.items() if not isinstance(value, str)}

    def _forward_for_export(*inner_args):
        positional = [inner_args[idx] for idx in positional_indices]
        kwargs = {key: inner_args[idx] for key, idx in kwarg_indices.items()}
        kwargs.update(kwarg_literals)
        return model(*positional, **kwargs)

    torch.onnx.export(
        _forward_for_export,
        export_args,
        output_path.as_posix(),
        opset_version=ONNX_CONFIG["opset"],
        do_constant_folding=True,
        input_names=list(effective_input_names),
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"[OK] ONNX model exported to {output_path}")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(RUNTIME_CONFIG["seed"])
    device = _select_device(RUNTIME_CONFIG["device"])

    model_cls = _import_symbol(MODEL_SPEC["target"])
    model: nn.Module = model_cls(**MODEL_SPEC.get("init_args", {}))
    if MODEL_SPEC.get("set_eval", True):
        model.eval()
    model.to(device)

    input_values, input_meta = _build_inputs(INPUT_SPECS, device)
    args, kwargs, ordered_input_names, ordered_input_values = _prepare_call_arguments(MODEL_SPEC, input_values)

    with torch.inference_mode():
        _ = model(*args, **kwargs)

    if RUNTIME_CONFIG.get("enable_torchinfo", True):
        _run_torchinfo(model, args, kwargs)
    if RUNTIME_CONFIG.get("enable_module_tree", True):
        _print_module_tree(model)

    gm: GraphModule = symbolic_trace(model)
    if ShapeProp is not None:
        ShapeProp(gm).propagate(*args, **kwargs)
    else:
        print("[Warn] ShapeProp is not available in this PyTorch version; node shapes may be missing.")

    _build_structure_graph(model, gm, input_meta, GRAPH_CONFIG, RUNTIME_CONFIG)
    _export_to_onnx(
        model,
        ordered_input_names,
        ordered_input_values,
        MODEL_SPEC.get("forward_args", []),
        MODEL_SPEC.get("forward_kwargs", {}),
        RUNTIME_CONFIG,
    )


if __name__ == "__main__":
    main()
