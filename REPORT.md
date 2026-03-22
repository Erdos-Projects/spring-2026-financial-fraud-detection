# Financial Fraud Detection Report

## Overview

This project studies financial fraud detection on two graph-based datasets:

1. The Elliptic Bitcoin transaction graph
2. The Elliptic++ transaction and address graph

The main goal was to compare a classical tabular baseline against graph neural network models and see whether richer graph structure improves illicit transaction detection. As requested, this draft excludes the logistic regression results and the `TestModels/` GCN results.

## 1. Problem Setting

Financial fraud is naturally relational. Fraudulent activity often appears as coordinated behavior across multiple accounts, addresses, or transactions rather than isolated events. That makes graph learning a good fit, because graph models can use both node features and neighborhood structure.

The data is also highly imbalanced. In both datasets, illicit nodes make up a small minority of the labeled samples, so accuracy alone is not a reliable metric. For this reason, the project emphasizes:

- Illicit-class F1
- Precision
- Recall
- ROC-AUC
- PR-AUC where available

## 2. Datasets

### 2.1 Elliptic Bitcoin Dataset

The Elliptic dataset contains:

- 203,769 transactions
- 165 usable node features
- 234,355 directed edges, symmetrized to 468,710 undirected edges in the GCN notebook
- Label distribution:
  - `unknown`: 157,205
  - `licit`: 42,019
  - `illicit`: 4,545

For the Elliptic experiments, the project used a temporal split:

- Training: steps 1 to 34
- Testing: steps 35 to 49

This is important because it reflects a realistic fraud setting where future transactions must be predicted from past behavior.

### 2.2 Elliptic++ Dataset

The Elliptic++ dataset expands the transaction graph with address-level information and wallet features.

Relevant statistics from the notebook:

- 203,769 transactions
- 182 baseline transaction features
- 234,355 transaction-to-transaction edges
- 837,124 transaction-to-address edges
- 2,868,964 address-to-address edges
- 1,268,260 wallet feature rows
- 157,205 unknown labels
- 42,019 licit labels
- 4,545 illicit labels

The Elliptic++ experiments also use a temporal split and additionally evaluate address-level supervision and multi-task learning.

## 3. Models Evaluated

### 3.1 XGBoost on Elliptic

This baseline uses transaction features only and treats fraud detection as a standard tabular binary classification problem.

### 3.2 GCN on Elliptic

This model uses a graph convolutional network with temporal train/test splitting and weighted loss to account for imbalance.

### 3.3 GCN Variants on Elliptic++

The Elliptic++ notebook compares several graph-based designs:

- Baseline GCN
- Wallet-upgrade GCN
- Directed heterogeneous GNN
- Multi-task directed heterogeneous GNN
- Address-pretraining followed by transaction fine-tuning

## 4. Results

### 4.1 Elliptic Results

| Model | Illicit F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.77 | 0.81 | 0.73 | 0.9230 | 0.7932 |
| GCN | 0.2410 | 0.1401 | 0.8596 | 0.8138 | N/A |

#### Interpretation

The XGBoost model performs strongly on the Elliptic dataset, especially in terms of illicit F1 and ROC-AUC. This suggests that the available transaction features are already highly informative and that a boosted tree model can exploit them effectively.

The GCN, by contrast, reaches very high recall for illicit nodes, but precision stays low. In practical terms, it finds many fraudulent transactions, but at the cost of many false positives. That pattern is typical on imbalanced graph fraud data when the default threshold is used.

### 4.2 Elliptic++ Results

| Model | Illicit F1 | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline GCN | 0.2147 | 0.1325 | 0.5660 | 0.7772 |
| Wallet-upgrade GCN | 0.2245 | 0.1379 | 0.6038 | 0.7769 |
| Directed hetero GNN | 0.2250 | 0.1636 | 0.3601 | 0.7920 |
| Address pretrain + finetune | 0.2379 | 0.1502 | 0.5708 | 0.7885 |
| Multi-task directed hetero GNN | 0.2918 | 0.1875 | 0.6572 | 0.8560 |

#### Interpretation

The multi-task directed heterogeneous GNN is the best-performing model on Elliptic++ by illicit F1 and ROC-AUC. It improves over the baseline GCN by:

- +0.0771 F1
- +0.0788 ROC-AUC

This is the clearest evidence in the project that richer graph structure and auxiliary supervision help fraud detection.

The address-pretraining approach also improves over the baseline, but it does not match the multi-task model. The directed heterogeneous model improves ROC-AUC, but its recall is lower than the baseline GCN, so it is more conservative in flagging illicit nodes.

The wallet-upgrade GCN gives only a small improvement over the baseline. That suggests feature enrichment helps, but not as much as modeling multiple node types and multiple tasks jointly.

## 5. Discussion

The results suggest three main conclusions:

1. **Tabular models are very strong when the feature space is rich.**
   - On Elliptic, XGBoost outperformed the GCN by a large margin on illicit F1 and ROC-AUC.

2. **Graph structure becomes more valuable as the dataset becomes richer and more heterogeneous.**
   - On Elliptic++, the best results came from the multi-task directed heterogeneous GNN, which could exploit transaction and address relations jointly.

3. **Class imbalance makes threshold choice critical.**
   - Several GNN variants achieved decent ROC-AUC but relatively low illicit precision.
   - That means the ranking quality was reasonable, but the default decision threshold was not always optimal for operational fraud screening.

## 6. Limitations

- The datasets are heavily imbalanced.
- Accuracy is not a useful headline metric because licit transactions dominate the test set.
- Some graph models may benefit from threshold tuning or calibration.
- Temporal drift may reduce performance if the fraud patterns in later time steps differ from earlier ones.
- The Elliptic++ notebook shows strong results for the multi-task model, but the evaluation is still based on the project’s current split and preprocessing choices.

## 7. Conclusion

Overall, the project shows that both classical machine learning and graph neural networks can be effective for fraud detection, but their strengths differ by dataset.

- On the Elliptic dataset, XGBoost was the strongest model.
- On the Elliptic++ dataset, the multi-task directed heterogeneous GNN was the strongest model.

This suggests that graph learning is most valuable when the problem has enough relational structure to exploit, while high-quality engineered features can still make tree-based methods very competitive.


