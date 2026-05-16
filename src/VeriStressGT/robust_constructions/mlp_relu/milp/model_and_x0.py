# robust_constructions/mlp_relu/milp/model_and_x0.py
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from VeriStressGT.utils.onnx_export import ExportConfig, export_pytorch_to_onnx


class MLP(nn.Module):
    def __init__(self, d: int = 10, h1: int = 20, h2: int = 20, C: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, C),
        )

    def forward(self, x):
        return self.net(x)


def export_onnx(model, d: int, path: str) -> None:
    """
    Export MILP construction models through the shared verifier-safe ONNX helper.

    This ensures:
      - legacy Torch ONNX exporter is used
      - requested opset is respected directly (instead of exporting at 18 and
        trying a failing version conversion)
      - model is saved as a single-file ONNX with embedded initializers
        so Marabou does not hit the external-data error
    """
    model.eval()

    cfg = ExportConfig(
        opset=13,
        do_constant_folding=False,
        dynamic_batch=False,
        use_external_data_format=False,
        keep_initializers_as_inputs=False,
        use_legacy_exporter=True,
    )

    dummy = torch.zeros(1, d, dtype=torch.float32).cpu().numpy()
    export_pytorch_to_onnx(
        model,
        output_path=path,
        input_shape=(1, d),
        config=cfg,
        example_input=dummy,
        input_name="input",
        output_name="logits",
    )


def sample_x0(model, d: int, tries: int = 1000, scale: float = 0.5) -> np.ndarray:
    model.eval()
    for _ in range(tries):
        x = torch.randn(1, d, dtype=torch.float32) * float(scale)
        with torch.no_grad():
            logits = model(x)
        if float(logits.abs().max().item()) > 1e-3:
            return x.squeeze(0).cpu().numpy()
    raise RuntimeError("Failed to find a non-degenerate x0")