#!/usr/bin/env python3
"""
Split the Elliptic Bitcoin dataset into temporal train/validation/test sets.

The Elliptic dataset is time-ordered, so this script uses the transaction
`time_step` column to avoid future leakage:

  - train:       time_step <= train_end_step
  - validation:  train_end_step < time_step <= val_end_step
  - test:        time_step > val_end_step

By default, the script keeps all transactions, including unknown labels, so the
resulting files remain useful for graph construction. Unknown labels are marked
with label = -1. Pass --known-only if you want to split only licit/illicit rows.

Outputs:
  - transactions_all.csv
  - train.csv
  - validation.csv
  - test.csv
  - edges.csv
  - split_summary.json

If --split-edges is provided, the script also writes:
  - edges_train.csv
  - edges_validation.csv
  - edges_test.csv
  - edges_cross_split.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "elliptic_bitcoin_data" / "elliptic_bitcoin_dataset"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "elliptic_bitcoin_splits"

LABEL_MAP = {
    "1": 1,
    "2": 0,
    "3": -1,
    "licit": 0,
    "illicit": 1,
    "unknown": -1,
}


def load_transactions(data_dir: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    features_path = data_dir / "elliptic_txs_features.csv"
    labels_path = data_dir / "elliptic_txs_classes.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    features_df = pd.read_csv(features_path, header=None)
    labels_df = pd.read_csv(labels_path)

    if features_df.shape[1] < 3:
        raise ValueError(
            "Expected elliptic_txs_features.csv to contain txId, time_step, and at least one feature column."
        )

    columns = ["txId", "time_step"] + [
        f"feature_{i}" for i in range(features_df.shape[1] - 2)
    ]
    features_df.columns = columns

    features_df["txId"] = pd.to_numeric(features_df["txId"], errors="raise").astype(np.int64)
    features_df["time_step"] = pd.to_numeric(features_df["time_step"], errors="raise").astype(np.int64)

    feature_columns = [c for c in features_df.columns if c not in {"txId", "time_step"}]
    features_df[feature_columns] = features_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features_df[feature_columns] = features_df[feature_columns].fillna(0.0)

    labels_df = labels_df.rename(columns={labels_df.columns[0]: "txId", labels_df.columns[1]: "class"})
    labels_df["txId"] = pd.to_numeric(labels_df["txId"], errors="raise").astype(np.int64)
    labels_df["class"] = labels_df["class"].astype(str)
    labels_df["label"] = labels_df["class"].map(LABEL_MAP).fillna(-1).astype(np.int64)

    transactions = features_df.merge(labels_df[["txId", "class", "label"]], on="txId", how="left")
    transactions["class"] = transactions["class"].fillna("unknown")
    transactions["label"] = transactions["label"].fillna(-1).astype(np.int64)
    transactions["is_known_label"] = transactions["label"].isin([0, 1])
    return transactions.sort_values(["time_step", "txId"]).reset_index(drop=True)


def load_edges(data_dir: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    edges_path = data_dir / "elliptic_txs_edgelist.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edge file: {edges_path}")

    edges_df = pd.read_csv(edges_path)
    if list(edges_df.columns[:2]) != ["txId1", "txId2"]:
        edges_df = edges_df.rename(columns={edges_df.columns[0]: "txId1", edges_df.columns[1]: "txId2"})

    edges_df["txId1"] = pd.to_numeric(edges_df["txId1"], errors="raise").astype(np.int64)
    edges_df["txId2"] = pd.to_numeric(edges_df["txId2"], errors="raise").astype(np.int64)
    return edges_df


def assign_split(time_step: pd.Series, train_end_step: int, val_end_step: int) -> pd.Series:
    import pandas as pd

    split = pd.Series(index=time_step.index, dtype="object")
    split.loc[time_step <= train_end_step] = "train"
    split.loc[(time_step > train_end_step) & (time_step <= val_end_step)] = "validation"
    split.loc[time_step > val_end_step] = "test"
    return split


def split_edges(edges: pd.DataFrame, tx_to_split: pd.Series) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    edge_splits = pd.DataFrame(
        {
            "txId1": edges["txId1"],
            "txId2": edges["txId2"],
            "split_1": edges["txId1"].map(tx_to_split),
            "split_2": edges["txId2"].map(tx_to_split),
        }
    )
    same_split = edge_splits["split_1"].notna() & edge_splits["split_1"].eq(edge_splits["split_2"])
    edge_splits["edge_split"] = np.where(
        same_split,
        edge_splits["split_1"],
        "cross_split",
    )
    return edge_splits.drop(columns=["split_1", "split_2"])


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_summary(all_transactions: pd.DataFrame, split_transactions: pd.DataFrame) -> dict:
    import pandas as pd

    summary = {
        "total_transactions": int(len(all_transactions)),
        "split_transaction_rows": int(len(split_transactions)),
        "time_step_min": int(all_transactions["time_step"].min()),
        "time_step_max": int(all_transactions["time_step"].max()),
        "label_counts": {
            str(k): int(v)
            for k, v in all_transactions["label"].value_counts(dropna=False).sort_index().items()
        },
        "split_counts": {
            str(k): int(v)
            for k, v in split_transactions["split"].value_counts(dropna=False).sort_index().items()
        },
        "known_label_counts_by_split": {},
    }

    for split_name, group in split_transactions.groupby("split", dropna=False):
        known_counts = group.loc[group["is_known_label"], "label"].value_counts().sort_index()
        summary["known_label_counts_by_split"][str(split_name)] = {
            str(k): int(v) for k, v in known_counts.items()
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the Elliptic Bitcoin dataset into temporal train/validation/test subsets."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-end-step", type=int, default=None)
    parser.add_argument("--val-end-step", type=int, default=None)
    parser.add_argument(
        "--known-only",
        action="store_true",
        help="Keep only licit/illicit rows in the output split files.",
    )
    parser.add_argument(
        "--split-edges",
        action="store_true",
        help="Also write edge lists partitioned by the split assignment of their endpoints.",
    )
    return parser.parse_args()


def compute_time_step_boundaries(
    transactions: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    import pandas as pd

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    step_counts = transactions.groupby("time_step").size().sort_index()
    cumulative = step_counts.cumsum() / step_counts.sum()
    time_steps = step_counts.index.to_list()

    train_target = train_ratio
    val_target = train_ratio + val_ratio

    train_pos = cumulative.searchsorted(train_target, side="left")
    val_pos = cumulative.searchsorted(val_target, side="left")

    train_pos = min(max(int(train_pos), 0), len(time_steps) - 3)
    val_pos = min(max(int(val_pos), train_pos + 1), len(time_steps) - 2)

    train_end_step = int(time_steps[train_pos])
    val_end_step = int(time_steps[val_pos])
    return train_end_step, val_end_step


def split_elliptic_bitcoin_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    train_end_step: int | None = None,
    val_end_step: int | None = None,
    known_only: bool = False,
    split_edges_flag: bool = False,
) -> dict:
    """Split Elliptic Bitcoin into temporal train/validation/test files."""
    import json

    if train_end_step is not None and train_end_step <= 0:
        raise ValueError("train_end_step must be positive.")
    if val_end_step is not None and train_end_step is not None and val_end_step <= train_end_step:
        raise ValueError("val_end_step must be greater than train_end_step.")

    all_transactions = load_transactions(data_dir)
    edges = load_edges(data_dir)

    if train_end_step is None or val_end_step is None:
        train_end_step, val_end_step = compute_time_step_boundaries(
            all_transactions, train_ratio=train_ratio, val_ratio=val_ratio
        )

    all_transactions["split"] = assign_split(
        all_transactions["time_step"], train_end_step, val_end_step
    )

    split_transactions = all_transactions.copy()
    if known_only:
        split_transactions = split_transactions[split_transactions["is_known_label"]].copy()

    all_transactions = all_transactions.sort_values(["split", "time_step", "txId"]).reset_index(drop=True)
    split_transactions = split_transactions.sort_values(["split", "time_step", "txId"]).reset_index(drop=True)
    tx_to_split = all_transactions.set_index("txId")["split"]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(all_transactions, output_dir / "transactions_all.csv")
    write_csv(split_transactions[split_transactions["split"] == "train"].copy(), output_dir / "train.csv")
    write_csv(
        split_transactions[split_transactions["split"] == "validation"].copy(),
        output_dir / "validation.csv",
    )
    write_csv(split_transactions[split_transactions["split"] == "test"].copy(), output_dir / "test.csv")
    write_csv(edges, output_dir / "edges.csv")

    if split_edges_flag:
        edge_splits = split_edges(edges, tx_to_split)
        write_csv(edge_splits[edge_splits["edge_split"] == "train"].copy(), output_dir / "edges_train.csv")
        write_csv(
            edge_splits[edge_splits["edge_split"] == "validation"].copy(),
            output_dir / "edges_validation.csv",
        )
        write_csv(edge_splits[edge_splits["edge_split"] == "test"].copy(), output_dir / "edges_test.csv")
        write_csv(
            edge_splits[edge_splits["edge_split"] == "cross_split"].copy(),
            output_dir / "edges_cross_split.csv",
        )

    summary = build_summary(all_transactions, split_transactions)
    summary.update(
        {
            "data_dir": str(data_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train_end_step": train_end_step,
            "val_end_step": val_end_step,
            "known_only": bool(known_only),
            "split_edges": bool(split_edges_flag),
        }
    )

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote split dataset to: {output_dir.resolve()}")
    print(f"  train rows:      {int((split_transactions['split'] == 'train').sum()):,}")
    print(f"  validation rows: {int((split_transactions['split'] == 'validation').sum()):,}")
    print(f"  test rows:       {int((split_transactions['split'] == 'test').sum()):,}")
    print(f"  edges:           {len(edges):,}")
    if split_edges_flag:
        print("  edge splits:     enabled")

    return summary


def main() -> None:
    args = parse_args()
    split_elliptic_bitcoin_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_end_step=args.train_end_step,
        val_end_step=args.val_end_step,
        known_only=args.known_only,
        split_edges_flag=args.split_edges,
    )


if __name__ == "__main__":
    main()
