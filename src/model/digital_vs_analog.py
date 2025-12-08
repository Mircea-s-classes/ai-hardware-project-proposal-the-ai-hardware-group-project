import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.utils import IOParameters, UpdateParameters
from aihwkit.simulator.configs.devices import ConstantStepDevice

# ---------------------------------------------------------
# Data loading (local MNIST, no downloads)
# ---------------------------------------------------------
data_dir = "data"

train_images_path = os.path.join(data_dir, "train-images.idx3-ubyte")
train_labels_path = os.path.join(data_dir, "train-labels.idx1-ubyte")
test_images_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(data_dir, "t10k-labels.idx1-ubyte")


def load_mnist_images(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(16), dtype=">i4")
        _, num_images, rows, cols = header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, rows, cols).astype("float32") / 255.0
    images = images[:, None, :, :]  # shape: (N, 1, 28, 28)
    return torch.from_numpy(images)


def load_mnist_labels(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=">i4")
        _, num_labels = header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    labels = data.astype("int64")  # shape: (N,)
    assert labels.shape[0] == num_labels
    return torch.from_numpy(labels)


train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

print("train_images:", train_images.shape)  # (60000, 1, 28, 28)
print("train_labels:", train_labels.shape)  # (60000,)
print("test_images:", test_images.shape)    # (10000, 1, 28, 28)
print("test_labels:", test_labels.shape)    # (10000,)

# ---------------------------------------------------------
# Common training setup
# ---------------------------------------------------------
batch_size = 64
hidden_units = 256
num_classes = 10
learning_rate = 0.01
num_epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.manual_seed(0)


# ---------------------------------------------------------
# RPU configs (hardware scenarios)
# ---------------------------------------------------------
rpu_fp = FloatingPointRPUConfig()  # near-ideal analog

rpu_mild = SingleRPUConfig(
    forward=IOParameters(out_noise=0.25),
    backward=IOParameters(out_noise=0.25),
    update=UpdateParameters(desired_bl=4),
    device=ConstantStepDevice()
)

rpu_harsh = SingleRPUConfig(
    forward=IOParameters(out_noise=0.5),
    backward=IOParameters(out_noise=0.5),
    update=UpdateParameters(desired_bl=3),
    device=ConstantStepDevice()
)


# ---------------------------------------------------------
# Model builder
# ---------------------------------------------------------
def build_model(kind: str, rpu_config=None) -> nn.Module:
    if kind == "digital":
        model = nn.Sequential(
            nn.Flatten(),                      # shape: (B, 1, 28, 28) -> (B, 784)
            nn.Linear(28 * 28, hidden_units),  # shape: (B, 784) -> (B, 256)
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)  # shape: (B, 256) -> (B, 10)
        )
    else:
        model = nn.Sequential(
            nn.Flatten(),                                      # shape: (B, 1, 28, 28) -> (B, 784)
            AnalogLinear(28 * 28, hidden_units, rpu_config=rpu_config),
            nn.ReLU(),
            AnalogLinear(hidden_units, num_classes, rpu_config=rpu_config)
        )
    return model.to(device)


# ---------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------
def train_and_eval(kind: str, rpu_config=None, log_batches: bool = False):
    model = build_model(kind, rpu_config)
    criterion = nn.CrossEntropyLoss()

    if kind == "digital":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
        optimizer.regroup_param_groups(model)

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)  # shape: (B, 1, 28, 28)
            labels = labels.to(device)  # shape: (B,)

            optimizer.zero_grad()
            outputs = model(images)     # shape: (B, 10)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            _, predicted = outputs.max(1)
            batch_correct = (predicted == labels).sum().item()

            running_loss += batch_loss * images.size(0)
            total += labels.size(0)
            correct += batch_correct

            if log_batches and (batch_idx % 100 == 0):
                batch_acc = batch_correct / labels.size(0)
                print(
                    f"[{kind}] Epoch {epoch+1}/{num_epochs} "
                    f"Batch {batch_idx+1}/{len(train_loader)} "
                    f"- loss: {batch_loss:.4f} - acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"[{kind}] Epoch {epoch+1}/{num_epochs} SUMMARY - "
              f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    end_time = time.perf_counter()
    train_time = end_time - start_time

    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= test_total
    test_acc = test_correct / test_total

    print(f"[{kind}] FINAL - train_time: {train_time:.2f}s "
          f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}")

    return {
        "train_time_sec": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


# ---------------------------------------------------------
# Run all four scenarios
# ---------------------------------------------------------
results = {}

results["digital"] = train_and_eval("digital", rpu_config=None)
results["analog_fp"] = train_and_eval("analog_fp", rpu_config=rpu_fp)
results["analog_mild"] = train_and_eval("analog_mild", rpu_config=rpu_mild)
results["analog_harsh"] = train_and_eval("analog_harsh", rpu_config=rpu_harsh)

print("\n=== SUMMARY (3 epochs, same architecture) ===")
for name, res in results.items():
    print(
        f"{name:12s} | time: {res['train_time_sec']:.2f}s "
        f"| test_loss: {res['test_loss']:.4f} "
        f"| test_acc: {res['test_acc']:.4f}"
    )
