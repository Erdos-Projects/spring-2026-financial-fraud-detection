# spring-2026-spec-proj
Team project: spring-2026-spec-proj

## setup

Create the conda environment defined in [environment.yml](/Users/kuijun/Desktop/spring-2026-spec-proj/environment.yml). It already includes PyTorch and graph libraries (`pyg`, `dgl`) needed for graph-based fraud modeling.

## project structure

## Data

The raw fraud dataset is the Elliptic Bitcoin transaction graph under [data/raw/elliptic_bitcoin_data/elliptic_bitcoin_dataset](/Users/kuijun/Desktop/spring-2026-spec-proj/data/raw/elliptic_bitcoin_data/elliptic_bitcoin_dataset).

To train a baseline graph neural network for illicit-vs-licit transaction detection:

```bash
python scripts/train_elliptic_gnn.py
```

Useful options:

```bash
python scripts/train_elliptic_gnn.py --epochs 100 --hidden-channels 128
python scripts/train_elliptic_gnn.py --directed
python scripts/train_elliptic_gnn.py --cpu
```

The script:
- Loads the raw node features, transaction edges, and class labels.
- Ignores `unknown` labels during supervised training.
- Builds train/validation/test splits on the known labels only.
- Trains a baseline GCN and prints validation plus test fraud-detection metrics.
