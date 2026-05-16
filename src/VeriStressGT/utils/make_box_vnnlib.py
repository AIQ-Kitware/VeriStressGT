from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _as_flat_float_array(center: Any, npz_key: str = "x") -> np.ndarray:
    """
    Normalize an input center into a 1D float64 NumPy array.

    Supported inputs:
      - path to a .npz file containing key `npz_key`
      - torch.Tensor-like object with detach().cpu().numpy()
      - NumPy array
      - Python list / tuple
    """
    if isinstance(center, (str, Path)):
        center_path = Path(center)
        if center_path.suffix != ".npz":
            raise ValueError(f"Expected .npz path, got: {center_path}")

        data = np.load(center_path)
        if npz_key not in data:
            raise KeyError(f".npz missing key '{npz_key}'. Keys: {list(data.keys())}")

        return np.asarray(data[npz_key], dtype=np.float64).reshape(-1)

    # torch.Tensor support without importing torch.
    if hasattr(center, "detach") and hasattr(center, "cpu") and hasattr(center, "numpy"):
        center = center.detach().cpu().numpy()

    return np.asarray(center, dtype=np.float64).reshape(-1)


def _append_robustness_violation(
    lines: list[str],
    *,
    num_outputs: int,
    label: int,
) -> None:
    """
    Append the VNN-LIB output constraint encoding the negation of robustness.

    The property is:
        exists j != label such that Y_j >= Y_label.

    Important parser compatibility detail:
    For binary classification, do NOT emit

        (assert (and (>= Y_1 Y_0)))

    because alpha-beta-CROWN's VNN-LIB parser can reject that top-level single
    `and`. Emit the atomic assertion directly instead:

        (assert (>= Y_1 Y_0))
    """
    num_outputs = int(num_outputs)
    label = int(label)

    if num_outputs < 1:
        raise ValueError("num_outputs must be >= 1")
    if not (0 <= label < num_outputs):
        raise ValueError(f"label={label} is out of range for num_outputs={num_outputs}")

    competitors = [j for j in range(num_outputs) if j != label]

    lines.append("")
    lines.append("; robustness violation: exists j != label with Y_j >= Y_label")

    if len(competitors) == 0:
        lines.append("(assert false)")
    elif len(competitors) == 1:
        j = competitors[0]
        lines.append(f"(assert (>= Y_{j} Y_{label}))")
    else:
        lines.append("(assert (or")
        for j in competitors:
            lines.append(f"    (and (>= Y_{j} Y_{label}))")
        lines.append("))")


def make_box_vnnlib(center, eps, out, num_outputs, label, npz_key: str = "x") -> None:
    """
    Write a VNN-LIB file for an L_inf robustness box.

    Args:
        center:
            Path to a .npz file with key `npz_key`, a torch.Tensor, a NumPy array,
            or a Python sequence. The center is flattened to shape [d].
        eps:
            Scalar L_inf radius.
        out:
            Output .vnnlib path.
        num_outputs:
            Number of output logits.
        label:
            Correct class label. The output assertion encodes the robustness
            violation: exists j != label with Y_j >= Y_label.
        npz_key:
            Key to use when `center` is a .npz path.
    """
    x = _as_flat_float_array(center, npz_key=npz_key)
    d = int(x.size)
    eps = float(eps)

    if d < 1:
        raise ValueError("center must contain at least one scalar")
    if eps < 0:
        raise ValueError("eps must be nonnegative")

    num_outputs = int(num_outputs)
    label = int(label)

    lo = x - eps
    hi = x + eps

    lines: list[str] = []

    # Declarations using the standard VNN-LIB naming convention.
    for i in range(d):
        lines.append(f"(declare-const X_{i} Real)")
    for j in range(num_outputs):
        lines.append(f"(declare-const Y_{j} Real)")

    lines.append("")
    lines.append("; input box")
    for i in range(d):
        lines.append(f"(assert (>= X_{i} {lo[i]}))")
        lines.append(f"(assert (<= X_{i} {hi[i]}))")

    _append_robustness_violation(lines, num_outputs=num_outputs, label=label)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write an L_inf box VNN-LIB robustness spec.")
    ap.add_argument("--center", required=True, help="Path to .npz file containing the center.")
    ap.add_argument("--npz-key", default="x", help="Key inside the .npz file.")
    ap.add_argument("--eps", type=float, required=True, help="L_inf radius.")
    ap.add_argument("--out", required=True, help="Output .vnnlib path.")
    ap.add_argument("--num_outputs", type=int, required=True, help="Number of output logits.")
    ap.add_argument("--label", type=int, required=True, help="Correct label index.")
    args = ap.parse_args()

    make_box_vnnlib(
        center=args.center,
        eps=args.eps,
        out=args.out,
        num_outputs=args.num_outputs,
        label=args.label,
        npz_key=args.npz_key,
    )


if __name__ == "__main__":
    main()
