# src/VeriStressGT/__init__.py
__all__ = [
    "__version__",
]

__version__ = "0.1.0"

# NOTE: We intentionally do NOT import torch, onnx, or any heavy deps here.
# Submodules use lazy imports via VeriStressGT._deps.require() so that
# `import VeriStressGT` succeeds even with only core deps installed.
