import os
import time
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.utils import IOParameters, UpdateParameters
from aihwkit.simulator.configs.devices import ConstantStepDevice

# ---------------------------------------------------------
# Data loading (MNIST IDX, local files)
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

print("train_images:", train_images.shape)  # shape: (60000, 1, 28, 28)
print("train_labels:", train_labels.shape)  # shape: (60000,)
print("test_images:", test_images.shape)  # shape: (10000, 1, 28, 28)
print("test_labels:", test_labels.shape)  # shape: (10000,)

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
# Helper: build model for a given RPU config
# ---------------------------------------------------------
def build_analog_model(rpu_config: SingleRPUConfig) -> nn.Module:
    model = nn.Sequential(
        nn.Flatten(),  # shape: (B, 1, 28, 28) -> (B, 784)
        AnalogLinear(28 * 28, hidden_units, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(hidden_units, num_classes, rpu_config=rpu_config)
    )
    return model.to(device)


# ---------------------------------------------------------
# Helper: train and evaluate one configuration
# ---------------------------------------------------------
def train_and_eval_config(forward_noise: float,
                          backward_noise: float,
                          desired_bl: int):
    rpu = SingleRPUConfig(
        forward=IOParameters(out_noise=forward_noise),
        backward=IOParameters(out_noise=backward_noise),
        update=UpdateParameters(desired_bl=desired_bl),
        device=ConstantStepDevice()
    )

    model = build_analog_model(rpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)  # shape: (B, 1, 28, 28)
            labels = labels.to(device)  # shape: (B,)

            optimizer.zero_grad()
            outputs = model(images)  # shape: (B, 10)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            _, predicted = outputs.max(1)
            batch_correct = (predicted == labels).sum().item()

            running_loss += batch_loss * images.size(0)
            total += labels.size(0)
            correct += batch_correct

        train_loss = running_loss / total
        train_acc = correct / total
        print(
            f"[fn={forward_noise:.2f}, bn={backward_noise:.2f}, bl={desired_bl}] "
            f"Epoch {epoch + 1}/{num_epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"
        )

    end_time = time.perf_counter()
    train_time = end_time - start_time

    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss_sum += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_loss_sum / test_total
    test_acc = test_correct / test_total

    print(
        f"[fn={forward_noise:.2f}, bn={backward_noise:.2f}, bl={desired_bl}] "
        f"FINAL - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f} "
        f"time: {train_time:.2f}s"
    )

    return train_loss, train_acc, test_loss, test_acc, train_time


# ---------------------------------------------------------
# Grid definition and CSV logging
# ---------------------------------------------------------
# forward/backward out_noise: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
forward_noises = [round(x * 0.2, 2) for x in range(0, 6)]
backward_noises = [round(x * 0.2, 2) for x in range(0, 6)]

# desired_bl: 1, 3, 5, 7  (step size 2)
desired_bls = [1, 3, 5, 7]

csv_path = "mnist_analog_sweep.csv"

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "forward_out_noise",
            "backward_out_noise",
            "desired_bl",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "train_time_sec",
        ],
    )
    writer.writeheader()

    total_configs = len(forward_noises) * len(backward_noises) * len(desired_bls)
    config_idx = 0

    for bl in desired_bls:
        for fn in forward_noises:
            for bn in backward_noises:
                config_idx += 1
                print(f"\n=== Config {config_idx}/{total_configs} "
                      f"(fn={fn:.2f}, bn={bn:.2f}, bl={bl}) ===")

                train_loss, train_acc, test_loss, test_acc, train_time = \
                    train_and_eval_config(fn, bn, bl)

                writer.writerow({
                    "forward_out_noise": fn,
                    "backward_out_noise": bn,
                    "desired_bl": bl,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "train_time_sec": train_time,
                })
                f.flush()

print(f"\nSaved sweep results to {csv_path}")
