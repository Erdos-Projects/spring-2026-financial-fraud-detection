#!/usr/bin/env python3
"""
Split the Elliptic++ dataset into temporal train/validation/test subsets.

This script follows the time-step convention used in the Elliptic++ notebooks:

  - train:      time_step <= train_end_step
  - validation: train_end_step < time_step <= val_end_step
  - test:       time_step > val_end_step

It creates split tables for both:
  - transaction rows from txs_features.csv / txs_classes.csv
  - wallet/address snapshot rows from wallets_features.csv / wallets_classes.csv

By default, the script preserves all rows, including unknown labels, so the
resulting files remain useful for graph construction. Unknown labels are mapped
to label = -1. Pass --known-only if you want the split files to contain only
licit/illicit rows.

Outputs:
  - transactions_all.csv
  - transactions_train.csv
  - transactions_validation.csv
  - transactions_test.csv
  - wallets_all.csv
  - wallets_train.csv
  - wallets_validation.csv
  - wallets_test.csv
  - txs_edgelist.csv
  - split_summary.json

If --split-tx-edges is provided, the script also writes:
  - txs_edges_train.csv
  - txs_edges_validation.csv
  - txs_edges_test.csv
  - txs_edges_cross_split.csv
  - TxAddr_edges_train.csv
  - TxAddr_edges_validation.csv
  - TxAddr_edges_test.csv
  - TxAddr_edges_cross_split.csv
  - AddrTx_edges_train.csv
  - AddrTx_edges_validation.csv
  - AddrTx_edges_test.csv
  - AddrTx_edges_cross_split.csv
  - AddrAddr_edges_train.csv
  - AddrAddr_edges_validation.csv
  - AddrAddr_edges_test.csv
  - AddrAddr_edges_cross_split.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "elliptic++dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "elliptic_plus_splits"

LABEL_MAP = {
    "1": 1,
    "2": 0,
    "3": -1,
    "licit": 0,
    "illicit": 1,
    "unknown": -1,
}


def clean_columns(columns: pd.Index) -> list[str]:
    cleaned: list[str] = []
    for col in columns:
        col = str(col).strip().replace(" ", "_").replace("-", "_").replace("/", "_")
        col = col.replace("(", "").replace(")", "").replace("__", "_")
        cleaned.append(col)
    return cleaned


def load_transactions(data_dir: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    features_path = data_dir / "txs_features.csv"
    labels_path = data_dir / "txs_classes.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing transaction features file: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing transaction labels file: {labels_path}")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    features_df.columns = clean_columns(features_df.columns)
    features_df = features_df.rename(columns={"Time_step": "time_step"})
    features_df["txId"] = pd.to_numeric(features_df["txId"], errors="raise").astype(np.int64)
    features_df["time_step"] = pd.to_numeric(features_df["time_step"], errors="raise").astype(np.int64)

    feature_columns = [c for c in features_df.columns if c not in {"txId", "time_step"}]
    features_df[feature_columns] = features_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features_df[feature_columns] = features_df[feature_columns].fillna(0.0)

    labels_df["txId"] = pd.to_numeric(labels_df["txId"], errors="raise").astype(np.int64)
    labels_df["class"] = labels_df["class"].astype(str)
    labels_df["label"] = labels_df["class"].map(LABEL_MAP).fillna(-1).astype(np.int64)

    transactions = features_df.merge(labels_df[["txId", "class", "label"]], on="txId", how="left")
    transactions["class"] = transactions["class"].fillna("unknown")
    transactions["label"] = transactions["label"].fillna(-1).astype(np.int64)
    transactions["is_known_label"] = transactions["label"].isin([0, 1])
    return transactions.sort_values(["time_step", "txId"]).reset_index(drop=True)


def load_wallets(data_dir: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    features_path = data_dir / "wallets_features.csv"
    labels_path = data_dir / "wallets_classes.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing wallet features file: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing wallet labels file: {labels_path}")

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    features_df.columns = clean_columns(features_df.columns)
    features_df = features_df.rename(columns={"Time_step": "time_step"})
    features_df["address"] = features_df["address"].astype(str)
    features_df["time_step"] = pd.to_numeric(features_df["time_step"], errors="raise").astype(np.int64)

    feature_columns = [c for c in features_df.columns if c not in {"address", "time_step"}]
    features_df[feature_columns] = features_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    features_df[feature_columns] = features_df[feature_columns].fillna(0.0)

    labels_df["address"] = labels_df["address"].astype(str)
    labels_df["class"] = labels_df["class"].astype(str)
    labels_df["label"] = labels_df["class"].map(LABEL_MAP).fillna(-1).astype(np.int64)

    wallets = features_df.merge(labels_df[["address", "class", "label"]], on="address", how="left")
    wallets["class"] = wallets["class"].fillna("unknown")
    wallets["label"] = wallets["label"].fillna(-1).astype(np.int64)
    wallets["is_known_label"] = wallets["label"].isin([0, 1])
    return wallets.sort_values(["time_step", "address"]).reset_index(drop=True)


def load_transaction_edges(data_dir: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    edges_path = data_dir / "txs_edgelist.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing transaction edge file: {edges_path}")

    edges_df = pd.read_csv(edges_path)
    if list(edges_df.columns[:2]) != ["txId1", "txId2"]:
        edges_df = edges_df.rename(columns={edges_df.columns[0]: "txId1", edges_df.columns[1]: "txId2"})

    edges_df["txId1"] = pd.to_numeric(edges_df["txId1"], errors="raise").astype(np.int64)
    edges_df["txId2"] = pd.to_numeric(edges_df["txId2"], errors="raise").astype(np.int64)
    return edges_df


def load_tx_addr_edges(data_dir: Path) -> pd.DataFrame:
    import pandas as pd

    edges_path = data_dir / "TxAddr_edgelist.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing TxAddr edge file: {edges_path}")

    edges_df = pd.read_csv(edges_path)
    edges_df = edges_df.rename(columns={edges_df.columns[0]: "txId", edges_df.columns[1]: "address"})
    return edges_df


def load_addr_tx_edges(data_dir: Path) -> pd.DataFrame:
    import pandas as pd

    edges_path = data_dir / "AddrTx_edgelist.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing AddrTx edge file: {edges_path}")

    edges_df = pd.read_csv(edges_path)
    edges_df = edges_df.rename(columns={edges_df.columns[0]: "address", edges_df.columns[1]: "txId"})
    return edges_df


def load_addr_addr_edges(data_dir: Path) -> pd.DataFrame:
    import pandas as pd

    edges_path = data_dir / "AddrAddr_edgelist.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing AddrAddr edge file: {edges_path}")

    edges_df = pd.read_csv(edges_path)
    edges_df = edges_df.rename(columns={edges_df.columns[0]: "src_address", edges_df.columns[1]: "dst_address"})
    return edges_df


def assign_split(time_step: pd.Series, train_end_step: int, val_end_step: int) -> pd.Series:
    import pandas as pd

    split = pd.Series(index=time_step.index, dtype="object")
    split.loc[time_step <= train_end_step] = "train"
    split.loc[(time_step > train_end_step) & (time_step <= val_end_step)] = "validation"
    split.loc[time_step > val_end_step] = "test"
    return split


def compute_address_split_map(wallets: pd.DataFrame, train_end_step: int, val_end_step: int):
    """Assign each address to a split based on its first observed time step."""
    first_seen = wallets.groupby("address")["time_step"].min()
    return assign_split(first_seen, train_end_step, val_end_step)


def split_edges_by_nodes(
    edges: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_split_map,
    right_split_map,
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    edge_splits = edges.copy()
    split_left = edge_splits[left_col].map(left_split_map)
    split_right = edge_splits[right_col].map(right_split_map)
    same_split = split_left.notna() & split_left.eq(split_right)
    edge_splits["edge_split"] = np.where(same_split, split_left, "cross_split")
    return edge_splits


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_summary(
    all_transactions: pd.DataFrame,
    split_transactions: pd.DataFrame,
    all_wallets: pd.DataFrame,
    split_wallets: pd.DataFrame,
) -> dict:
    import pandas as pd

    summary = {
        "transactions": {
            "total_rows": int(len(all_transactions)),
            "split_rows": int(len(split_transactions)),
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
        },
        "wallets": {
            "total_rows": int(len(all_wallets)),
            "split_rows": int(len(split_wallets)),
            "time_step_min": int(all_wallets["time_step"].min()),
            "time_step_max": int(all_wallets["time_step"].max()),
            "label_counts": {
                str(k): int(v)
                for k, v in all_wallets["label"].value_counts(dropna=False).sort_index().items()
            },
            "split_counts": {
                str(k): int(v)
                for k, v in split_wallets["split"].value_counts(dropna=False).sort_index().items()
            },
        },
        "known_label_counts_by_split": {
            "transactions": {},
            "wallets": {},
        },
    }

    for split_name, group in split_transactions.groupby("split", dropna=False):
        counts = group.loc[group["is_known_label"], "label"].value_counts().sort_index()
        summary["known_label_counts_by_split"]["transactions"][str(split_name)] = {
            str(k): int(v) for k, v in counts.items()
        }

    for split_name, group in split_wallets.groupby("split", dropna=False):
        counts = group.loc[group["is_known_label"], "label"].value_counts().sort_index()
        summary["known_label_counts_by_split"]["wallets"][str(split_name)] = {
            str(k): int(v) for k, v in counts.items()
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the Elliptic++ dataset into temporal train/validation/test subsets."
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
        help="Keep only licit/illicit rows in the split files.",
    )
    parser.add_argument(
        "--split-tx-edges",
        action="store_true",
        help="Also write transaction edge lists partitioned by transaction split.",
    )
    return parser.parse_args()


def compute_time_step_boundaries(
    transactions: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    step_counts = transactions.groupby("time_step").size().sort_index()
    cumulative = step_counts.cumsum() / step_counts.sum()
    time_steps = step_counts.index.to_list()

    train_pos = cumulative.searchsorted(train_ratio, side="left")
    val_pos = cumulative.searchsorted(train_ratio + val_ratio, side="left")

    train_pos = min(max(int(train_pos), 0), len(time_steps) - 3)
    val_pos = min(max(int(val_pos), train_pos + 1), len(time_steps) - 2)

    return int(time_steps[train_pos]), int(time_steps[val_pos])


def split_elliptic_plus_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    train_end_step: int | None = None,
    val_end_step: int | None = None,
    known_only: bool = False,
    split_tx_edges_flag: bool = False,
) -> dict:
    """Split Elliptic++ into temporal train/validation/test files."""
    import json

    if train_end_step is not None and train_end_step <= 0:
        raise ValueError("train_end_step must be positive.")
    if val_end_step is not None and train_end_step is not None and val_end_step <= train_end_step:
        raise ValueError("val_end_step must be greater than train_end_step.")

    all_transactions = load_transactions(data_dir)
    all_wallets = load_wallets(data_dir)
    tx_edges = load_transaction_edges(data_dir)
    tx_addr_edges = load_tx_addr_edges(data_dir)
    addr_tx_edges = load_addr_tx_edges(data_dir)
    addr_addr_edges = load_addr_addr_edges(data_dir)

    if train_end_step is None or val_end_step is None:
        train_end_step, val_end_step = compute_time_step_boundaries(
            all_transactions, train_ratio=train_ratio, val_ratio=val_ratio
        )

    all_transactions["split"] = assign_split(
        all_transactions["time_step"], train_end_step, val_end_step
    )
    all_wallets["split"] = assign_split(
        all_wallets["time_step"], train_end_step, val_end_step
    )
    address_split_map = compute_address_split_map(all_wallets, train_end_step, val_end_step)

    split_transactions = all_transactions.copy()
    split_wallets = all_wallets.copy()
    if known_only:
        split_transactions = split_transactions[split_transactions["is_known_label"]].copy()
        split_wallets = split_wallets[split_wallets["is_known_label"]].copy()

    all_transactions = all_transactions.sort_values(["split", "time_step", "txId"]).reset_index(drop=True)
    split_transactions = split_transactions.sort_values(["split", "time_step", "txId"]).reset_index(drop=True)
    all_wallets = all_wallets.sort_values(["split", "time_step", "address"]).reset_index(drop=True)
    split_wallets = split_wallets.sort_values(["split", "time_step", "address"]).reset_index(drop=True)

    tx_to_split = all_transactions.set_index("txId")["split"]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(all_transactions, output_dir / "transactions_all.csv")
    write_csv(
        split_transactions[split_transactions["split"] == "train"].copy(),
        output_dir / "transactions_train.csv",
    )
    write_csv(
        split_transactions[split_transactions["split"] == "validation"].copy(),
        output_dir / "transactions_validation.csv",
    )
    write_csv(
        split_transactions[split_transactions["split"] == "test"].copy(),
        output_dir / "transactions_test.csv",
    )

    write_csv(all_wallets, output_dir / "wallets_all.csv")
    write_csv(split_wallets[split_wallets["split"] == "train"].copy(), output_dir / "wallets_train.csv")
    write_csv(
        split_wallets[split_wallets["split"] == "validation"].copy(),
        output_dir / "wallets_validation.csv",
    )
    write_csv(split_wallets[split_wallets["split"] == "test"].copy(), output_dir / "wallets_test.csv")

    write_csv(tx_edges, output_dir / "txs_edgelist.csv")

    if split_tx_edges_flag:
        tx_edge_splits = split_edges_by_nodes(tx_edges, "txId1", "txId2", tx_to_split, tx_to_split)
        write_csv(
            tx_edge_splits[tx_edge_splits["edge_split"] == "train"].copy(),
            output_dir / "txs_edges_train.csv",
        )
        write_csv(
            tx_edge_splits[tx_edge_splits["edge_split"] == "validation"].copy(),
            output_dir / "txs_edges_validation.csv",
        )
        write_csv(
            tx_edge_splits[tx_edge_splits["edge_split"] == "test"].copy(),
            output_dir / "txs_edges_test.csv",
        )
        write_csv(
            tx_edge_splits[tx_edge_splits["edge_split"] == "cross_split"].copy(),
            output_dir / "txs_edges_cross_split.csv",
        )

        tx_addr_edge_splits = split_edges_by_nodes(
            tx_addr_edges,
            "txId",
            "address",
            tx_to_split,
            address_split_map,
        )
        write_csv(
            tx_addr_edge_splits[tx_addr_edge_splits["edge_split"] == "train"].copy(),
            output_dir / "TxAddr_edges_train.csv",
        )
        write_csv(
            tx_addr_edge_splits[tx_addr_edge_splits["edge_split"] == "validation"].copy(),
            output_dir / "TxAddr_edges_validation.csv",
        )
        write_csv(
            tx_addr_edge_splits[tx_addr_edge_splits["edge_split"] == "test"].copy(),
            output_dir / "TxAddr_edges_test.csv",
        )
        write_csv(
            tx_addr_edge_splits[tx_addr_edge_splits["edge_split"] == "cross_split"].copy(),
            output_dir / "TxAddr_edges_cross_split.csv",
        )

        addr_tx_edge_splits = split_edges_by_nodes(
            addr_tx_edges,
            "address",
            "txId",
            address_split_map,
            tx_to_split,
        )
        write_csv(
            addr_tx_edge_splits[addr_tx_edge_splits["edge_split"] == "train"].copy(),
            output_dir / "AddrTx_edges_train.csv",
        )
        write_csv(
            addr_tx_edge_splits[addr_tx_edge_splits["edge_split"] == "validation"].copy(),
            output_dir / "AddrTx_edges_validation.csv",
        )
        write_csv(
            addr_tx_edge_splits[addr_tx_edge_splits["edge_split"] == "test"].copy(),
            output_dir / "AddrTx_edges_test.csv",
        )
        write_csv(
            addr_tx_edge_splits[addr_tx_edge_splits["edge_split"] == "cross_split"].copy(),
            output_dir / "AddrTx_edges_cross_split.csv",
        )

        addr_addr_edge_splits = split_edges_by_nodes(
            addr_addr_edges,
            "src_address",
            "dst_address",
            address_split_map,
            address_split_map,
        )
        write_csv(
            addr_addr_edge_splits[addr_addr_edge_splits["edge_split"] == "train"].copy(),
            output_dir / "AddrAddr_edges_train.csv",
        )
        write_csv(
            addr_addr_edge_splits[addr_addr_edge_splits["edge_split"] == "validation"].copy(),
            output_dir / "AddrAddr_edges_validation.csv",
        )
        write_csv(
            addr_addr_edge_splits[addr_addr_edge_splits["edge_split"] == "test"].copy(),
            output_dir / "AddrAddr_edges_test.csv",
        )
        write_csv(
            addr_addr_edge_splits[addr_addr_edge_splits["edge_split"] == "cross_split"].copy(),
            output_dir / "AddrAddr_edges_cross_split.csv",
        )

    summary = build_summary(all_transactions, split_transactions, all_wallets, split_wallets)
    summary.update(
        {
            "data_dir": str(data_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "train_end_step": train_end_step,
            "val_end_step": val_end_step,
            "known_only": bool(known_only),
            "split_tx_edges": bool(split_tx_edges_flag),
        }
    )

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote split dataset to: {output_dir.resolve()}")
    print(f"  transactions train:      {int((split_transactions['split'] == 'train').sum()):,}")
    print(f"  transactions validation: {int((split_transactions['split'] == 'validation').sum()):,}")
    print(f"  transactions test:       {int((split_transactions['split'] == 'test').sum()):,}")
    print(f"  wallets train:           {int((split_wallets['split'] == 'train').sum()):,}")
    print(f"  wallets validation:      {int((split_wallets['split'] == 'validation').sum()):,}")
    print(f"  wallets test:            {int((split_wallets['split'] == 'test').sum()):,}")
    print(f"  tx edges:                {len(tx_edges):,}")
    if split_tx_edges_flag:
        print("  tx edge splits:          enabled")

    return summary


def main() -> None:
    args = parse_args()
    split_elliptic_plus_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_end_step=args.train_end_step,
        val_end_step=args.val_end_step,
        known_only=args.known_only,
        split_tx_edges_flag=args.split_tx_edges,
    )


if __name__ == "__main__":
    main()
