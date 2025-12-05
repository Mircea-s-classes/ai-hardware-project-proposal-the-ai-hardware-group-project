"""Evaluate the MNIST baseline with AIHWKIT analog layers."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import InferenceRPUConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model.baseline_mnist import BaselineNet, get_loaders, resolve_device

DATA_DIR = REPO_ROOT / "data"
DEFAULT_BASELINE = DATA_DIR / "baseline.pt"
METRICS_PATH = DATA_DIR / "analog_metrics.csv"


def make_rpu_config(out_noise: Optional[float]) -> InferenceRPUConfig:
    """Build a simple inference RPU config with an optional output noise level."""
    rpu_config = InferenceRPUConfig()
    if out_noise is not None:
        rpu_config.forward.out_noise = out_noise
    return rpu_config


def build_analog_model(
    baseline_path: Path,
    device: torch.device,
    rpu_config: InferenceRPUConfig,
) -> nn.Module:
    baseline = BaselineNet()
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline weights not found at {baseline_path}. "
            "Run src/model/baseline_mnist.py first."
        )
    state_dict = torch.load(baseline_path, map_location="cpu")
    baseline.load_state_dict(state_dict)
    analog_model = convert_to_analog(baseline, rpu_config)
    analog_model.to(device)
    analog_model.eval()
    return analog_model


def estimate_macs(
    model: nn.Module, device: torch.device, input_size=(1, 1, 28, 28)
) -> int:
    """Rough MAC count for conv/linear layers on a dummy input."""
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _: torch.Tensor, out: torch.Tensor) -> None:
        nonlocal macs
        batch = out.shape[0]
        out_elems = out[0].numel()
        kh, kw = module.kernel_size
        groups = module.groups if module.groups else 1
        macs += batch * out_elems * (module.in_channels // groups) * kh * kw

    def linear_hook(module: nn.Linear, inp: torch.Tensor, out: torch.Tensor) -> None:
        nonlocal macs
        batch = out.shape[0]
        macs += batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        dummy = torch.zeros(*input_size, device=device)
        model(dummy)

    for handle in handles:
        handle.remove()
    return macs


def evaluate_analog(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mac_energy: float,
    macs_per_image: Optional[int] = None,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    latency_s = time.perf_counter() - start

    test_loss = total_loss / total
    test_acc = total_correct / total

    if macs_per_image is None:
        macs_per_image = estimate_macs(model, device)
    energy_per_image_j = macs_per_image * mac_energy
    energy_dataset_j = energy_per_image_j * total

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "latency_s": latency_s,
        "macs_per_image": macs_per_image,
        "energy_j_per_image": energy_per_image_j,
        "energy_j_dataset": energy_dataset_j,
    }


def write_metrics(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "config_name",
        "baseline_path",
        "batch_size",
        "out_noise",
        "mac_energy_j",
        "test_loss",
        "test_acc",
        "latency_s",
        "macs_per_image",
        "energy_j_per_image",
        "energy_j_dataset",
        "device",
    ]
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline weights with AIHWKIT analog layers."
    )
    parser.add_argument("--config-name", type=str, default="default")
    parser.add_argument("--out-noise", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps|auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (set to 0 to avoid macOS semaphore issues).",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=str(DEFAULT_BASELINE),
        help="Path to baseline .pt file.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=str(METRICS_PATH),
        help="CSV to append analog run metrics.",
    )
    parser.add_argument(
        "--mac-energy",
        type=float,
        default=1e-12,
        help="Energy per MAC (J). Used for a simple energy proxy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "mps":
        print("[analog] MPS is not supported for AIHWKIT; falling back to CPU.")
        device = torch.device("cpu")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rpu_config = make_rpu_config(args.out_noise)
    analog_model = build_analog_model(
        Path(args.baseline_path), device, rpu_config
    )
    _, test_loader = get_loaders(
        batch_size=64, test_batch_size=args.batch_size, num_workers=args.num_workers
    )
    metrics = evaluate_analog(
        analog_model,
        test_loader,
        device,
        mac_energy=args.mac_energy,
    )

    metrics_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_name": args.config_name,
        "baseline_path": str(args.baseline_path),
        "batch_size": args.batch_size,
        "out_noise": args.out_noise,
        "mac_energy_j": args.mac_energy,
        "test_loss": metrics["test_loss"],
        "test_acc": metrics["test_acc"],
        "latency_s": round(metrics["latency_s"], 4),
        "macs_per_image": metrics["macs_per_image"],
        "energy_j_per_image": metrics["energy_j_per_image"],
        "energy_j_dataset": metrics["energy_j_dataset"],
        "device": str(device),
    }
    write_metrics(metrics_row, Path(args.metrics_path))

    print(
        f"{args.config_name}: acc={metrics['test_acc']:.4f}, "
        f"loss={metrics['test_loss']:.4f}, latency_s={metrics['latency_s']:.3f}, "
        f"energy_j_per_image={metrics['energy_j_per_image']:.3e}"
    )
    print(f"Logged metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
