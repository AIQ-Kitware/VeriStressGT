# src/VeriStressGT/utils/onnx_export.py
"""
ONNX export utilities.

Heavy deps (torch, onnx, onnxruntime) are imported lazily so that
``import VeriStressGT`` works even with only core deps installed.
The actual import happens when a function here is *called*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ExportConfig:
    """ONNX export configuration."""
    opset: int = 18
    do_constant_folding: bool = False
    dynamic_batch: bool = False
    use_external_data_format: bool = False
    keep_initializers_as_inputs: bool = False
    # Critical for verifier compatibility: use legacy exporter by default.
    # The new dynamo exporter may silently export at a newer opset and then
    # attempt a failing version-conversion pass.
    use_legacy_exporter: bool = True


def parse_shape(shape_csv: str) -> Tuple[int, ...]:
    """Parse a shape like '1,3,224,224' into a tuple."""
    parts = [p.strip() for p in shape_csv.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty shape")
    return tuple(int(p) for p in parts)


def _load_onnx():
    from VeriStressGT._deps import require
    return require("onnx")


def _maybe_embed_external_data(onnx_path: str) -> None:
    """
    Re-save an ONNX model as a single-file model with embedded initializers.

    This matters for verifiers like Marabou whose command-line ONNX support
    does not implement external tensor data locations.
    """
    onnx = _load_onnx()
    path = Path(onnx_path)

    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=False,
    )

    # Clean up common sidecar names if present.
    sidecars = [
        path.with_suffix(path.suffix + ".data"),
        path.parent / f"{path.name}.data",
    ]
    for extra in sidecars:
        try:
            if extra.exists():
                extra.unlink()
        except Exception:
            pass


def export_pytorch_to_onnx(
    model,
    output_path: str,
    input_shape: Sequence[int],
    config: ExportConfig = ExportConfig(),
    *,
    example_input: Optional[np.ndarray] = None,
    input_name: str = "input",
    output_name: str = "output",
) -> None:
    """Export a PyTorch model to ONNX."""
    from VeriStressGT._deps import require
    torch = require("torch")

    model.eval()

    if example_input is None:
        x_np = np.random.randn(*input_shape).astype(np.float32)
    else:
        x_np = example_input.astype(np.float32, copy=False)

    x = torch.from_numpy(x_np)

    output_path = str(Path(output_path).expanduser().resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if config.dynamic_batch:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    export_kwargs = dict(
        model=model,
        args=x,
        f=output_path,
        export_params=True,
        opset_version=config.opset,
        do_constant_folding=config.do_constant_folding,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=config.keep_initializers_as_inputs,
    )

    # Prefer the legacy exporter for verifier-facing exports.
    # This is the key fix for your failed "request opset 13 but actually get 18"
    # behavior seen in the logs.
    if config.use_legacy_exporter:
        try:
            torch.onnx.export(
                **export_kwargs,
                dynamo=False,
            )
        except TypeError:
            # Older torch versions may not expose the dynamo kwarg.
            torch.onnx.export(**export_kwargs)
    else:
        # Best-effort support for the newer exporter path.
        try:
            torch.onnx.export(
                **export_kwargs,
                dynamo=True,
            )
        except TypeError:
            torch.onnx.export(**export_kwargs)

    # Normalize final artifact format.
    if not config.use_external_data_format:
        _maybe_embed_external_data(output_path)


def check_onnx_loadable(path: str) -> None:
    """Basic sanity check: can onnx / onnxruntime load the exported model?"""
    from VeriStressGT._deps import require
    onnx = require("onnx")
    onnx.load(path)

    try:
        import onnxruntime as ort
        ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except ImportError:
        pass
    except Exception:
        pass