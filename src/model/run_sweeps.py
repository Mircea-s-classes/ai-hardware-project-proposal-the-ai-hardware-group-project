"""Run a small set of AIHWKIT analog configs and log results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aihwkit.simulator.configs import InferenceRPUConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model.analog_mnist import (  # noqa: E402
    DATA_DIR,
    DEFAULT_BASELINE,
    build_analog_model,
    evaluate_analog,
    make_rpu_config,
    write_metrics,
)
from model.baseline_mnist import get_loaders, resolve_device  # noqa: E402

RUNS_CSV = DATA_DIR / "analog_runs.csv"
FIG_PATH = DATA_DIR / "figures" / "acc_energy.png"


def parse_noise_list(arg: str) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for part in arg.split(","):
        token = part.strip()
        if not token or token.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(token))
    return values


def config_name(noise: Optional[float]) -> str:
    if noise is None:
        return "ideal"
    return f"noise_{noise}"


def run_config(
    out_noise: Optional[float],
    baseline_path: Path,
    device,
    batch_size: int,
    mac_energy: float,
    metrics_path: Path,
    test_loader,
) -> None:
    rpu_config: InferenceRPUConfig = make_rpu_config(out_noise)
    analog_model = build_analog_model(baseline_path, device, rpu_config)
    metrics = evaluate_analog(
        analog_model,
        test_loader,
        device,
        mac_energy=mac_energy,
    )
    row = {
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "config_name": config_name(out_noise),
        "baseline_path": str(baseline_path),
        "batch_size": batch_size,
        "out_noise": out_noise,
        "mac_energy_j": mac_energy,
        "test_loss": metrics["test_loss"],
        "test_acc": metrics["test_acc"],
        "latency_s": round(metrics["latency_s"], 4),
        "macs_per_image": metrics["macs_per_image"],
        "energy_j_per_image": metrics["energy_j_per_image"],
        "energy_j_dataset": metrics["energy_j_dataset"],
        "device": str(device),
    }
    write_metrics(row, metrics_path)
    print(
        f"{row['config_name']}: acc={row['test_acc']:.4f}, "
        f"energy/image={row['energy_j_per_image']:.3e}, "
        f"latency_s={row['latency_s']:.3f}"
    )


def plot_results(csv_path: Path, fig_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["energy_j_per_image"], df["test_acc"], color="tab:blue")
    for _, row in df.iterrows():
        ax.annotate(
            row["config_name"],
            (row["energy_j_per_image"], row["test_acc"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Energy per image (J)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("AIHWKIT analog sweeps")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved plot to {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small AIHWKIT sweep.")
    parser.add_argument(
        "--out-noise-list",
        type=str,
        default="0,0.02,0.05",
        help="Comma list of output noise values (use 'none' for ideal).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps|auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 to avoid macOS semaphore issues).",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=str(DEFAULT_BASELINE),
        help="Path to baseline.pt weights.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=str(RUNS_CSV),
        help="CSV to write sweep results.",
    )
    parser.add_argument(
        "--mac-energy",
        type=float,
        default=1e-12,
        help="Energy per MAC (J) for proxy calculations.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the accuracy vs energy plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "mps":
        print("[sweep] MPS is not supported for AIHWKIT; falling back to CPU.")
        device = torch.device("cpu")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _, test_loader = get_loaders(
        batch_size=64,
        test_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    noises = parse_noise_list(args.out_noise_list)
    for noise in noises:
        run_config(
            out_noise=noise,
            baseline_path=Path(args.baseline_path),
            device=device,
            batch_size=args.batch_size,
            mac_energy=args.mac_energy,
            metrics_path=Path(args.metrics_path),
            test_loader=test_loader,
        )

    if not args.no_plot:
        plot_results(Path(args.metrics_path), FIG_PATH)


if __name__ == "__main__":
    main()
