"""Train a digital MNIST baseline and save weights/metrics."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
MODEL_PATH = DATA_DIR / "baseline.pt"
METRICS_PATH = DATA_DIR / "baseline_metrics.csv"


class BaselineNet(nn.Module):
    """Small CNN used for the MNIST baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(
    batch_size: int, test_batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, total_correct / total


def write_metrics(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "epochs",
        "batch_size",
        "test_batch_size",
        "lr",
        "momentum",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "duration_s",
        "device",
        "model_path",
    ]
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a digital MNIST baseline.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers. Use 0 if you hit macOS semaphore issues.",
    )
    parser.add_argument(
        "--save-path", type=str, default=str(MODEL_PATH), help="Where to store weights."
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=str(METRICS_PATH),
        help="CSV to append training metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_loaders(
        args.batch_size, args.test_batch_size, num_workers=args.num_workers
    )
    model = BaselineNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    start = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )
    duration_s = time.perf_counter() - start

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")

    metrics_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "duration_s": round(duration_s, 2),
        "device": str(device),
        "model_path": str(save_path),
    }
    write_metrics(metrics_row, Path(args.metrics_path))
    print(f"Logged metrics to {args.metrics_path}")


if __name__ == "__main__":
    main()
