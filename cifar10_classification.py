# cifar10_classification.py
import random
from itertools import product
from typing import Tuple, List, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------
# Device setup
# ---------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------
# Data loading
# ---------------------------------------------
def load_cifar10(sample_data: bool = False, batch_size: int = 128):
    train_dataset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
    )

    if sample_data:
        train_dataset, _ = random_split(
            train_dataset, [int(0.1 * len(train_dataset)), int(0.9 * len(train_dataset))]
        )

    train_dataset, val_dataset = random_split(
        train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# ---------------------------------------------
# Models
# ---------------------------------------------
def linear_model(input_dim: int = 3072) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 10)
    ).to(DEVICE)

def non_linear_model(M: int, input_dim: int = 3072) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, M),
        nn.ReLU(),
        nn.Linear(M, 10)
    ).to(DEVICE)

class ConvModel(nn.Module):
    def __init__(self, M: int, k: int, N: int):
        super().__init__()
        self.conv = nn.Conv2d(3, M, kernel_size=k)
        self.pool = nn.MaxPool2d(kernel_size=N)
        self.flatten = nn.Flatten()
        # Calculate flattened size after conv + pooling
        self.fc_input_size = M * ((32 - k + 1) // N) * ((32 - k + 1) // N)
        self.fc = nn.Linear(self.fc_input_size, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def cnn_model(M: int, k: int, N: int) -> nn.Module:
    return ConvModel(M, k, N).to(DEVICE)

# ---------------------------------------------
# Training and evaluation
# ---------------------------------------------
def train(
    model: nn.Module,
    optimizer: SGD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int
) -> Tuple[List[float], List[float], List[float], List[float]]:
    loss_fn = nn.CrossEntropyLoss()
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for _ in range(epochs):
        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (pred.argmax(1) == y).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / (len(train_loader.dataset)))

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                val_acc += (pred.argmax(1) == y).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader.dataset))

    return train_losses, train_accs, val_losses, val_accs

def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            test_acc += (pred.argmax(1) == y).sum().item()
    return test_loss / len(loader), test_acc / len(loader.dataset)

# ---------------------------------------------
# Hyperparameter search
# ---------------------------------------------
def parameter_search_fc(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_fn: Callable[[int], nn.Module],
    epochs: int = 20,
    num_iter: int = 10
) -> List[Tuple[float, float, int]]:
    results = []
    for _ in range(num_iter):
        lr = 10 ** random.uniform(-6, -1)
        M = random.randint(128, 512)
        print(f"Testing lr={lr:.6f}, M={M}")
        model = model_fn(M)
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        _, _, _, val_accs = train(model, optimizer, train_loader, val_loader, epochs)
        results.append((max(val_accs), lr, M))
    results.sort(reverse=True)
    return results[:3]

def parameter_search_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20
) -> List[Tuple[float, int, int, int]]:
    results = []
    M_options, k_options, N_options = [64, 256, 512], [3, 5], [2, 3, 6]
    for M, k, N in product(M_options, k_options, N_options):
        print(f"Testing CNN M={M}, k={k}, N={N}")
        model = cnn_model(M, k, N)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        _, _, _, val_accs = train(model, optimizer, train_loader, val_loader, epochs)
        results.append((max(val_accs), M, k, N))
    results.sort(reverse=True)
    return results[:3]

# ---------------------------------------------
# Plot results
# ---------------------------------------------
def plot_accuracies(top_results, title="Training/Validation Accuracy"):
    plt.figure(figsize=(10, 6))
    for i, result in enumerate(top_results):
        train_acc = result[-2]
        val_acc = result[-1]
        epochs = range(1, len(train_acc)+1)
        plt.plot(epochs, train_acc, label=f"Config {i+1} Train")
        plt.plot(epochs, val_acc, linestyle="--", label=f"Config {i+1} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------------------------
# Main execution
# ---------------------------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_cifar10(sample_data=False)

    # Example: Non-linear fully connected
    top_fc = parameter_search_fc(train_loader, val_loader, non_linear_model)
    print("Top 3 FC configurations:", top_fc)

    # Example: CNN
    top_cnn = parameter_search_cnn(train_loader, val_loader)
    print("Top 3 CNN configurations:", top_cnn)
