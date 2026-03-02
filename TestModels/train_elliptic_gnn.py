"""
Train a baseline graph neural network on the Elliptic Bitcoin dataset.

This script uses the raw CSVs already present in:
  data/raw/elliptic_bitcoin_data/elliptic_bitcoin_dataset

Label mapping:
  - licit -> 0
  - illicit -> 1
  - unknown -> ignored during supervised training/evaluation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "elliptic_bitcoin_data" / "elliptic_bitcoin_dataset"
)
LABEL_MAP = {
    "2": 0,
    "1": 1,
    "licit": 0,
    "illicit": 1,
    "unknown": -1,
}


@dataclass
class DatasetBundle:
    data: Data
    tx_ids: pd.Series
    known_mask: np.ndarray


class FraudGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def load_elliptic_dataset(data_dir: Path, make_undirected: bool = True) -> DatasetBundle:
    features_path = data_dir / "elliptic_txs_features.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"
    labels_path = data_dir / "elliptic_txs_classes.csv"

    if not all(path.exists() for path in (features_path, edges_path, labels_path)):
        missing = [str(path) for path in (features_path, edges_path, labels_path) if not path.exists()]
        raise FileNotFoundError(f"Missing required dataset files: {missing}")

    features_df = pd.read_csv(features_path, header=None)
    edges_df = pd.read_csv(edges_path)
    labels_df = pd.read_csv(labels_path)

    tx_ids = features_df.iloc[:, 0].astype(np.int64)
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids.tolist())}

    # Keep every model feature column except the transaction id.
    x = torch.tensor(features_df.iloc[:, 1:].to_numpy(dtype=np.float32), dtype=torch.float32)

    label_series = labels_df.set_index("txId")["class"].astype(str).map(LABEL_MAP)
    y_np = tx_ids.map(label_series).fillna(-1).astype(np.int64).to_numpy()
    y = torch.tensor(y_np, dtype=torch.long)

    src = edges_df["txId1"].map(id_to_idx)
    dst = edges_df["txId2"].map(id_to_idx)
    valid_edges = src.notna() & dst.notna()
    edge_index = torch.tensor(
        np.vstack(
            [
                src[valid_edges].to_numpy(dtype=np.int64),
                dst[valid_edges].to_numpy(dtype=np.int64),
            ]
        ),
        dtype=torch.long,
    )

    if make_undirected:
        edge_index = to_undirected(edge_index)

    known_mask = y_np >= 0
    data = Data(x=x, edge_index=edge_index, y=y)
    return DatasetBundle(data=data, tx_ids=tx_ids, known_mask=known_mask)


def build_masks(
    labels: np.ndarray,
    known_mask: np.ndarray,
    train_size: float,
    val_size: float,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    known_indices = np.where(known_mask)[0]
    known_labels = labels[known_indices]

    train_idx, held_out_idx = train_test_split(
        known_indices,
        train_size=train_size,
        random_state=seed,
        stratify=known_labels,
    )

    held_out_labels = labels[held_out_idx]
    relative_val_size = val_size / (1.0 - train_size)
    val_idx, test_idx = train_test_split(
        held_out_idx,
        train_size=relative_val_size,
        random_state=seed,
        stratify=held_out_labels,
    )

    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(labels.shape[0], dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def evaluate(model: nn.Module, data: Data, mask: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[mask].argmax(dim=1).cpu().numpy()
        targets = data.y[mask].cpu().numpy()

    return {
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
    }


def train_model(
    data: Data,
    hidden_channels: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    device: torch.device,
) -> tuple[FraudGCN, dict[str, float]]:
    model = FraudGCN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_targets = data.y[data.train_mask]
    class_counts = torch.bincount(train_targets, minlength=2).clamp_min(1).float()
    class_weights = (class_counts.sum() / (2.0 * class_counts)).to(device)

    best_state = None
    best_val_f1 = -1.0
    best_metrics: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, data, data.val_mask)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_precision={val_metrics['precision']:.4f} | "
                f"val_recall={val_metrics['recall']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def print_test_report(model: nn.Module, data: Data) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        targets = data.y[data.test_mask].cpu().numpy()

    print("\nTest metrics")
    print(f"F1:        {f1_score(targets, preds, zero_division=0):.4f}")
    print(f"Precision: {precision_score(targets, preds, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(targets, preds, zero_division=0):.4f}")
    print("\nClassification report")
    print(
        classification_report(
            targets,
            preds,
            labels=[0, 1],
            target_names=["licit", "illicit"],
            zero_division=0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline GCN for fraud detection on the Elliptic graph."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_size <= 0 or args.val_size <= 0 or args.train_size + args.val_size >= 1:
        raise ValueError("train-size and val-size must be > 0 and sum to less than 1.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bundle = load_elliptic_dataset(
        data_dir=args.data_dir,
        make_undirected=not args.directed,
    )
    train_mask, val_mask, test_mask = build_masks(
        labels=bundle.data.y.cpu().numpy(),
        known_mask=bundle.known_mask,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    bundle.data.train_mask = train_mask
    bundle.data.val_mask = val_mask
    bundle.data.test_mask = test_mask

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    known_count = int(bundle.known_mask.sum())
    fraud_count = int((bundle.data.y.cpu().numpy() == 1).sum())
    print(f"Loaded graph with {bundle.data.num_nodes:,} nodes and {bundle.data.num_edges:,} edges.")
    print(f"Known labels: {known_count:,} | illicit labels: {fraud_count:,}")
    print(
        f"Split sizes | train={int(train_mask.sum()):,} "
        f"val={int(val_mask.sum()):,} test={int(test_mask.sum()):,}"
    )
    print(f"Training on device: {device}")

    model, best_val_metrics = train_model(
        data=bundle.data,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
    )
    bundle.data = bundle.data.to(device)

    print(
        "\nBest validation metrics | "
        f"f1={best_val_metrics.get('f1', 0.0):.4f} | "
        f"precision={best_val_metrics.get('precision', 0.0):.4f} | "
        f"recall={best_val_metrics.get('recall', 0.0):.4f}"
    )
    print_test_report(model, bundle.data)


if __name__ == "__main__":
    main()
