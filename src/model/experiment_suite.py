"""End-to-end experiment runner for the class project.

- Trains the digital MNIST baseline (unless weights already exist or --force-train).
- Evaluates baseline latency/energy proxies.
- Runs a small AIHWKIT analog sweep.
- Logs all results to CSV/JSON/plot for presentation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from model.baseline_mnist import (  # noqa: E402
    BaselineNet,
    evaluate as eval_digital,
    get_loaders,
    resolve_device,
    train_one_epoch,
)
from model.analog_mnist import (  # noqa: E402
    DEFAULT_BASELINE,
    DATA_DIR,
    METRICS_PATH as ANALOG_CSV,
    build_analog_model,
    evaluate_analog,
    make_rpu_config,
    write_metrics as write_analog_csv,
    estimate_macs,
)

RUNS_CSV = DATA_DIR / "analog_runs.csv"
SUMMARY_JSON = DATA_DIR / "summary.json"
SUMMARY_MD = DATA_DIR / "summary.md"


def parse_noise_list(arg: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for part in arg.split(","):
        token = part.strip()
        if not token or token.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(token))
    return values


def train_baseline(
    device: torch.device,
    epochs: int,
    batch_size: int,
    test_batch_size: int,
    lr: float,
    momentum: float,
    save_path: Path,
    num_workers: int,
) -> nn.Module:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders(
        batch_size, test_batch_size, num_workers=num_workers
    )
    model = BaselineNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = eval_digital(model, test_loader, criterion, device)
        print(
            f"[baseline] epoch {epoch + 1}/{epochs} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[baseline] saved weights to {save_path}")
    return model


def evaluate_baseline(
    model: nn.Module,
    eval_device: torch.device,
    test_batch_size: int,
    mac_energy: float,
    num_workers: int,
) -> Dict[str, float]:
    _, test_loader = get_loaders(
        batch_size=64, test_batch_size=test_batch_size, num_workers=num_workers
    )
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_analog(model.to(eval_device), test_loader, eval_device, mac_energy)
    test_loss, test_acc = eval_digital(model, test_loader, criterion, eval_device)
    metrics["test_loss"] = test_loss
    metrics["test_acc"] = test_acc
    return metrics


def run_analog_sweep(
    noises: List[Optional[float]],
    baseline_path: Path,
    device: torch.device,
    batch_size: int,
    mac_energy: float,
    metrics_path: Path,
    num_workers: int,
) -> List[Dict]:
    analog_device = device
    if analog_device.type == "mps":
        print("[suite] AIHWKIT not supported on MPS; using CPU for analog.")
        analog_device = torch.device("cpu")
    _, test_loader = get_loaders(
        batch_size=64, test_batch_size=batch_size, num_workers=num_workers
    )
    results = []
    for noise in noises:
        config = make_rpu_config(noise)
        analog_model = build_analog_model(baseline_path, analog_device, config)
        metrics = evaluate_analog(
            analog_model,
            test_loader,
            analog_device,
            mac_energy=mac_energy,
        )
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": "ideal" if noise in (None, 0) else f"noise_{noise}",
            "baseline_path": str(baseline_path),
            "batch_size": batch_size,
            "out_noise": noise,
            "mac_energy_j": mac_energy,
            "test_loss": metrics["test_loss"],
            "test_acc": metrics["test_acc"],
            "latency_s": round(metrics["latency_s"], 4),
            "macs_per_image": metrics["macs_per_image"],
            "energy_j_per_image": metrics["energy_j_per_image"],
            "energy_j_dataset": metrics["energy_j_dataset"],
            "device": str(analog_device),
        }
        write_analog_csv(row, metrics_path)
        print(
            f"[analog {row['config_name']}] acc={row['test_acc']:.4f} "
            f"energy/img={row['energy_j_per_image']:.3e} "
            f"latency_s={row['latency_s']:.3f}"
        )
        results.append(row)
    return results


def summarize(
    baseline: Dict[str, float],
    analog_runs: List[Dict],
    acc_target: float = 0.975,
    energy_improve_target: float = 0.40,
    latency_improve_target: float = 0.20,
) -> Dict:
    baseline_energy = baseline["energy_j_per_image"]
    baseline_latency = baseline["latency_s"]

    def improvements(run: Dict) -> Dict[str, float]:
        energy_gain = 1 - (run["energy_j_per_image"] / baseline_energy)
        latency_gain = 1 - (run["latency_s"] / baseline_latency)
        return {"energy_gain": energy_gain, "latency_gain": latency_gain}

    best = None
    best_acc = -1.0
    for run in analog_runs:
        gains = improvements(run)
        run["energy_gain"] = gains["energy_gain"]
        run["latency_gain"] = gains["latency_gain"]
        if run["test_acc"] > best_acc:
            best_acc = run["test_acc"]
            best = run

    passing = [
        r
        for r in analog_runs
        if r["test_acc"] >= acc_target
        and (1 - r["energy_j_per_image"] / baseline_energy) >= energy_improve_target
        and (1 - r["latency_s"] / baseline_latency) >= latency_improve_target
    ]
    winner = passing[0] if passing else None

    summary = {
        "baseline": baseline,
        "analog_runs": analog_runs,
        "best_accuracy": best,
        "passes_targets": winner,
        "targets": {
            "acc_target": acc_target,
            "energy_improve_target": energy_improve_target,
            "latency_improve_target": latency_improve_target,
        },
    }
    return summary


def write_summary(summary: Dict) -> None:
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    lines = []
    baseline = summary["baseline"]
    lines.append("# Experiment Summary\n")
    lines.append("## Baseline (digital)\n")
    lines.append(
        f"- test_acc: {baseline['test_acc']:.4f}\n"
        f"- latency_s: {baseline['latency_s']:.4f}\n"
        f"- energy_j_per_image: {baseline['energy_j_per_image']:.3e}\n"
        f"- macs_per_image: {baseline['macs_per_image']}\n"
    )
    lines.append("## Analog runs\n")
    for run in summary["analog_runs"]:
        lines.append(
            f"- {run['config_name']}: acc={run['test_acc']:.4f}, "
            f"energy_gain={(1 - run['energy_j_per_image'] / baseline['energy_j_per_image']):.2%}, "
            f"latency_gain={(1 - run['latency_s'] / baseline['latency_s']):.2%}"
        )
    if summary["passes_targets"]:
        winner = summary["passes_targets"]
        lines.append("\n## Meets targets")
        lines.append(
            f"- {winner['config_name']} meets accuracy, energy, and latency targets."
        )
    else:
        lines.append("\n## Meets targets")
        lines.append("- No analog config met all targets yet.")
    SUMMARY_MD.write_text("\n".join(lines))
    print(f"[summary] wrote {SUMMARY_JSON} and {SUMMARY_MD}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline + AIHWKIT analog sweep and log summary."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps|auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 to avoid macOS semaphore/leak issues).",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=str(DEFAULT_BASELINE),
        help="Where to store or read baseline weights.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain baseline even if weights already exist.",
    )
    parser.add_argument(
        "--mac-energy",
        type=float,
        default=1e-12,
        help="Energy per MAC (J) for proxy calculations.",
    )
    parser.add_argument(
        "--out-noise-list",
        type=str,
        default="0,0.02,0.05",
        help="Comma list of output noise values (use 'none' for ideal).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    baseline_path = Path(args.baseline_path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if baseline_path.exists() and not args.force_train:
        print(f"[baseline] using existing weights at {baseline_path}")
        model = BaselineNet()
        model.load_state_dict(torch.load(baseline_path, map_location="cpu"))
        model.to(device)
    else:
        model = train_baseline(
            device,
            args.epochs,
            args.batch_size,
            args.test_batch_size,
            args.lr,
            args.momentum,
            baseline_path,
            args.num_workers,
        )

    baseline_metrics = evaluate_baseline(
        model, device, args.test_batch_size, args.mac_energy, args.num_workers
    )
    print(
        f"[baseline] acc={baseline_metrics['test_acc']:.4f} "
        f"latency_s={baseline_metrics['latency_s']:.3f} "
        f"energy/img={baseline_metrics['energy_j_per_image']:.3e}"
    )

    noises = parse_noise_list(args.out_noise_list)
    analog_runs = run_analog_sweep(
        noises=noises,
        baseline_path=baseline_path,
        device=device,
        batch_size=args.test_batch_size,
        mac_energy=args.mac_energy,
        metrics_path=RUNS_CSV,
        num_workers=args.num_workers,
    )

    summary = summarize(baseline_metrics, analog_runs)
    write_summary(summary)


if __name__ == "__main__":
    main()
