"""Minimal AIHWKIT analog training demo (single AnalogLinear layer)."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD


def main() -> None:
    device = torch.device("cpu")  # AIHWKIT supports CPU (no MPS)

    # Tiny toy dataset.
    x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]]).to(device)
    y = Tensor([[1.0, 0.5], [0.7, 0.3]]).to(device)

    # Single analog layer.
    model = AnalogLinear(4, 2).to(device)

    # Analog-aware optimizer.
    opt = AnalogSGD(model.parameters(), lr=0.1)
    opt.regroup_param_groups(model)

    for epoch in range(10):
        opt.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
