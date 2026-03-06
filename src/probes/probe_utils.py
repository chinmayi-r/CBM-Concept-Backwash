# src/probes/probe_utils.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class ProbeMetrics:
    train_acc: float
    val_acc: float
    best_val_acc: float


class LinearProbe(nn.Module):
    """
    A single linear layer used as a probing classifier.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> ProbeMetrics:
    """
    Train a simple linear probe for classification.

    Returns: ProbeMetrics with train/val accuracies.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = x_train.shape[1]
    model = LinearProbe(in_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(x_train, y_train.long())
    val_ds = TensorDataset(x_val, y_val.long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    final_train_acc = 0.0
    final_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tqdm(train_loader, desc=f"Probe epoch {epoch}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate at end of epoch
        model.eval()
        with torch.no_grad():
            train_correct, train_total = 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += yb.size(0)
            final_train_acc = train_correct / train_total

            val_correct, val_total = 0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)
            final_val_acc = val_correct / val_total

        best_val_acc = max(best_val_acc, final_val_acc)

    return ProbeMetrics(
        train_acc=float(final_train_acc),
        val_acc=float(final_val_acc),
        best_val_acc=float(best_val_acc),
    )
