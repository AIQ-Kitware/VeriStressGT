"""
Embedded Projection for Provably Robust Neural Network Instances
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from dataclasses import dataclass
from VeriStressGT.utils.make_box_vnnlib import make_box_vnnlib
from VeriStressGT.utils.onnx_export import export_pytorch_to_onnx

CONSTRUCTION_NAME = "mlp_relu.embedded_projection"

@dataclass
class EmbeddedProjectionConfig:
    """Configuration for embedded projection layer"""
    x0: np.ndarray
    epsilon: float
    eta: float = 0.0
    c: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=np.float32)
        if self.c is None:
            self.c = self.x0.copy() + torch.randn_like(torch.from_numpy(self.x0)).numpy() * 2.0
        else:
            self.c = np.asarray(self.c, dtype=np.float32)
        
        if self.x0.shape != self.c.shape:
            raise ValueError(f"x0 shape {self.x0.shape} must match c shape {self.c.shape}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.eta < 0:
            raise ValueError(f"eta must be non-negative, got {self.eta}")


class EmbeddedProjection(nn.Module):
    """Neural network layer that projects an epsilon-ball to a constant"""
    
    def __init__(self, config: EmbeddedProjectionConfig):
        super().__init__()
        
        self.d = config.x0.size
        self.epsilon = config.epsilon
        self.eta = config.eta
        
        # construct diff of relus with weights to cancel out at end of projection
        alpha = config.x0 - config.epsilon - config.eta
        beta = config.x0 + config.epsilon + config.eta
        
        W1 = np.zeros((2 * self.d, self.d), dtype=np.float32)
        b1 = np.zeros(2 * self.d, dtype=np.float32)
        
        for i in range(self.d):
            W1[2 * i, i] = -1.0
            b1[2 * i] = alpha[i]
            W1[2 * i + 1, i] = 1.0
            b1[2 * i + 1] = -beta[i]
        
        W2 = np.zeros((self.d, 2 * self.d), dtype=np.float32)
        for i in range(self.d):
            W2[i, 2 * i] = 1.0
            W2[i, 2 * i + 1] = 1.0
        
        b2 = config.c.copy()
        
        self.linear1 = nn.Linear(self.d, 2 * self.d, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * self.d, self.d, bias=True)
        
        with torch.no_grad():
            self.linear1.weight.copy_(torch.from_numpy(W1))
            self.linear1.bias.copy_(torch.from_numpy(b1))
            self.linear2.weight.copy_(torch.from_numpy(W2))
            self.linear2.bias.copy_(torch.from_numpy(b2))
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.register_buffer('x0', torch.from_numpy(config.x0))
        self.register_buffer('c', torch.from_numpy(config.c))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        
        h = self.relu(self.linear1(x))
        out = self.linear2(h)
        
        if squeeze:
            out = out.squeeze(0)
        return out


class RobustNetwork(nn.Module):
    """Wrapper combining EmbeddedProjection with an arbitrary downstream network"""
    
    def __init__(self, projection: EmbeddedProjection, downstream: nn.Module):
        super().__init__()
        self.projection = projection
        self.downstream = downstream
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return self.downstream(projected)


class MLP(nn.Module):
    """Configurable MLP with variable depth and width"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_and_export_instance(
    model_factory,
    x0: np.ndarray,
    epsilon: float,
    eta: float,
    num_outputs: int,
    onnx_path: Union[str, Path],
    vnnlib_path: Union[str, Path],
    label: Optional[int] = None,
) -> dict:
    onnx_path = Path(onnx_path)
    vnnlib_path = Path(vnnlib_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    vnnlib_path.parent.mkdir(parents=True, exist_ok=True)

    x0 = np.asarray(x0, dtype=np.float32).flatten()
    input_dim = x0.size

    downstream = model_factory()
    config = EmbeddedProjectionConfig(x0=x0, epsilon=epsilon, eta=eta)
    projection = EmbeddedProjection(config)
    robust_model = RobustNetwork(projection, downstream)
    robust_model.eval()

    if label is None:
        with torch.no_grad():
            out0 = robust_model(torch.from_numpy(x0).unsqueeze(0))
            label = int(out0.argmax(dim=-1).item())

    export_pytorch_to_onnx(robust_model, str(onnx_path), (1, input_dim))
    # try:
    #     simplify_onnx_inplace(onnx_path, log_path=onnx_path.with_suffix(".onnxsim.log"))
    # except subprocess.CalledProcessError as e:
    #     print(f"[ERROR] onnxsim failed with exit code {e.returncode}")
        
    make_box_vnnlib(center=x0, eps=epsilon, out=str(vnnlib_path), num_outputs=num_outputs, label=label)

    return {
        "onnx_path": str(onnx_path),
        "vnnlib_path": str(vnnlib_path),
        "x0": x0,
        "epsilon": float(epsilon),
        "eta": float(eta),
        "label": int(label),
        "is_robust": True,
    }

import subprocess
import sys
from pathlib import Path

def simplify_onnx_inplace(onnx_path: Union[str, Path], log_path: Optional[Union[str, Path]] = None) -> None:
    onnx_path = str(onnx_path)
    cmd = [sys.executable, "-m", "onnxsim", onnx_path, onnx_path]  # in-place: input == output

    if log_path is None:
        # print simplify output to console
        subprocess.run(cmd, check=True)
    else:
        log_path = str(log_path)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

def get_hidden_dims(args) -> List[int]:
    """Determine hidden dimensions from args"""
    
    # Predefined scales
    scales = {
        'tiny':   [1000, 1000],
        'small':  [1000,1000,1000],
        'medium': [1000, 1000, 1000, 1000],
        'large':  [1000, 1000, 1000, 1000, 1000, 1000],
        'xlarge': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    }
    
    if args.scale is not None:
        return scales[args.scale]
    elif args.hidden_dims is not None:
        return args.hidden_dims
    elif args.wide_layers > 0:
        # Easy scaling: 64 -> [wide_width]*N -> 32
        return [64] + [args.wide_width] * args.wide_layers + [32]
    else:
        return [args.width] * args.num_layers

def add_args(p: argparse.ArgumentParser):
    p.add_argument('--input-dim', type=int, default=100,
                   help='Input dimension')
    p.add_argument('--output-dim', type=int, default=10,
                   help='Output dimension (number of classes)')
    p.add_argument('--hidden-dims', type=int, nargs='+', default=None,
                   help='Explicit hidden layer dimensions (e.g., --hidden-dims 64 256 256 64)')
    p.add_argument('--num-layers', type=int, default=5,
                   help='Number of hidden layers (used if --hidden-dims not specified)')
    p.add_argument('--width', type=int, default=256,
                   help='Width of each hidden layer (used if --hidden-dims not specified)')
    p.add_argument('--wide-layers', type=int, default=0,
                   help='Number of 1000-neuron layers to add (creates: 64 -> [1000]*N -> 32 -> output)')
    p.add_argument('--wide-width', type=int, default=1000,
                   help='Width of the wide layers (default: 1000)')
    p.add_argument('--scale', type=str, default=None,
                   choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
                   help='Predefined architecture scale (overrides other options)')
    p.add_argument('--epsilon', type=float, default=0.2,
                   help='Perturbation radius')
    p.add_argument('--eta', type=float, default=0.0,
                   help='Additional margin for projection layer')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')

def run(args):
    # Ensure output parents exist (driver will pass onnx_path/vnnlib_path)
    Path(args.onnx_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vnnlib_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(args.seed))
    torch.manual_seed(int(args.seed))

    hidden_dims = get_hidden_dims(args)

    print("=" * 60)
    print("Embedded Projection - Provably Robust Instance Generator")
    print("=" * 60)
    print(f"Input dim:     {args.input_dim}")
    print(f"Output dim:    {args.output_dim}")
    print(f"Hidden dims:   {hidden_dims}")
    print(f"Num layers:    {len(hidden_dims)}")
    print(f"Total neurons: {sum(hidden_dims)}")
    print(f"Epsilon:       {args.epsilon}")
    print(f"Eta:           {args.eta}")
    print("=" * 60)

    x0 = rng.randn(int(args.input_dim)).astype(np.float32) * 0.2

    def model_factory():
        return MLP(int(args.input_dim), int(args.output_dim), hidden_dims)

    # IMPORTANT: write exactly to args.onnx_path / args.vnnlib_path
    result = create_and_export_instance(
        model_factory=model_factory,
        x0=x0,
        epsilon=float(args.epsilon),
        eta=float(args.eta),
        num_outputs=int(args.output_dim),
        onnx_path=args.onnx_path,
        vnnlib_path=args.vnnlib_path,
        label=getattr(args, "label", None),
    )

    print(f"Wrote ONNX:   {result['onnx_path']}")
    print(f"Wrote VNNLIB: {result['vnnlib_path']}")
    return result




def _main():
    parser = argparse.ArgumentParser(description="Create provable embedded projection instance")
    add_args(parser)
    args = parser.parse_args()
    if args.onnx_path is None or args.vnnlib_path is None:
        raise ValueError("Standalone mode requires --onnx_path and --vnnlib_path")
    run(args)

if __name__ == "__main__":
    _main()
