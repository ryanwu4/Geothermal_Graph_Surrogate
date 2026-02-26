from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import h5py
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GCNConv, global_add_pool, global_mean_pool


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraphScaler:
    def __init__(
        self,
        whiten: bool,
        pca_components: int | None,
        log_target: bool,
    ) -> None:
        self.node_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.whiten = whiten
        self.pca_components = pca_components
        self.log_target = log_target
        self.node_pca: PCA | None = None

    def _target_forward(self, y: np.ndarray) -> np.ndarray:
        if self.log_target:
            return np.log1p(np.clip(y, a_min=0.0, a_max=None))
        return y

    def _target_inverse(self, y: np.ndarray) -> np.ndarray:
        if self.log_target:
            return np.expm1(y)
        return y

    def fit(self, graphs: list[Data]) -> None:
        node_train = np.concatenate([g.x.cpu().numpy() for g in graphs if g.x.shape[0] > 0], axis=0)
        global_train = np.concatenate([g.global_attr.cpu().numpy() for g in graphs], axis=0)
        y_raw = np.array([float(g.y.item()) for g in graphs], dtype=np.float32).reshape(-1, 1)

        node_scaled = self.node_scaler.fit_transform(node_train)
        if self.whiten:
            self.node_pca = PCA(n_components=self.pca_components, whiten=True, random_state=42)
            self.node_pca.fit(node_scaled)

        self.global_scaler.fit(global_train)

        y_transformed = self._target_forward(y_raw)
        self.target_scaler.fit(y_transformed)

    def transform_graph(self, graph: Data) -> Data:
        x_np = graph.x.cpu().numpy()
        x_scaled = self.node_scaler.transform(x_np)
        if self.node_pca is not None:
            x_scaled = self.node_pca.transform(x_scaled)

        global_np = graph.global_attr.cpu().numpy()
        global_scaled = self.global_scaler.transform(global_np)

        y_raw = np.array([[float(graph.y.item())]], dtype=np.float32)
        y_transformed = self._target_forward(y_raw)
        y_scaled = self.target_scaler.transform(y_transformed).reshape(-1)

        transformed = Data(
            x=torch.tensor(x_scaled, dtype=torch.float32),
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            global_attr=torch.tensor(global_scaled, dtype=torch.float32),
            y=torch.tensor(y_scaled, dtype=torch.float32),
            case_id=graph.case_id,
        )
        return transformed

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        y_unscaled = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))
        y_raw = self._target_inverse(y_unscaled)
        return y_raw.reshape(-1)


def build_knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = coords.shape[0]
    if n_nodes <= 1:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, n_nodes - 1)

    source_nodes: list[int] = []
    target_nodes: list[int] = []
    edge_distances: list[float] = []
    for i in range(n_nodes):
        nbrs = np.argpartition(dist[i], kth=k_eff - 1)[:k_eff]
        for j in nbrs:
            d = float(dist[i, j])
            source_nodes.append(i)
            target_nodes.append(int(j))
            edge_distances.append(d)

            source_nodes.append(int(j))
            target_nodes.append(i)
            edge_distances.append(d)

    edge_index = np.array([source_nodes, target_nodes], dtype=np.int64)

    edge_dist = np.array(edge_distances, dtype=np.float32)
    sigma = float(np.median(edge_dist)) + 1e-6
    edge_weight = np.exp(-edge_dist / sigma).astype(np.float32)
    return edge_index, edge_weight


def add_god_node(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = node_features.shape[0]
    if n_nodes == 0:
        return node_features, edge_index, edge_weight

    god_feature = np.array(
        [
            float(np.mean(node_features[:, 0])),
            float(np.mean(node_features[:, 1])),
            float(np.mean(node_features[:, 2])),
            0.5,
            float(np.sum(node_features[:, 4])),
            float(np.mean(node_features[:, 5])),
        ],
        dtype=np.float32,
    ).reshape(1, -1)

    node_features_aug = np.vstack([node_features, god_feature]).astype(np.float32)
    god_idx = n_nodes

    god_src = np.concatenate([np.full(n_nodes, god_idx, dtype=np.int64), np.arange(n_nodes, dtype=np.int64)])
    god_dst = np.concatenate([np.arange(n_nodes, dtype=np.int64), np.full(n_nodes, god_idx, dtype=np.int64)])
    god_edge_index = np.vstack([god_src, god_dst]).astype(np.int64)
    god_edge_weight = np.ones(god_edge_index.shape[1], dtype=np.float32)

    if edge_index.size == 0:
        edge_index_aug = god_edge_index
        edge_weight_aug = god_edge_weight
    else:
        edge_index_aug = np.concatenate([edge_index, god_edge_index], axis=1)
        edge_weight_aug = np.concatenate([edge_weight, god_edge_weight], axis=0)

    return node_features_aug, edge_index_aug, edge_weight_aug


def load_graphs(h5_path: Path, knn_k: int, use_god_node: bool) -> tuple[list[Data], np.ndarray]:
    graphs: list[Data] = []
    skipped_empty = 0
    all_targets: list[float] = []

    with h5py.File(h5_path, "r") as handle:
        for case_id in sorted(handle.keys()):
            group = handle[case_id]
            wells = group["wells"][:]
            production_total = group["outputs"]["field_energy_production_total"][:]

            n_wells = len(wells)
            if n_wells == 0:
                skipped_empty += 1
                continue

            x = wells["x"].astype(np.float32)
            y = wells["y"].astype(np.float32)
            depth = wells["depth"].astype(np.float32)
            inj_rate = wells["inj_rate"].astype(np.float32)
            is_injector = (inj_rate > 0).astype(np.float32)
            abs_inj_rate = np.abs(inj_rate)

            coords = np.stack([x, y, depth], axis=1)
            node_features = np.stack([x, y, depth, is_injector, inj_rate, abs_inj_rate], axis=1)
            edge_index, edge_weight = build_knn_edges(coords, k=knn_k)
            if use_god_node:
                node_features, edge_index, edge_weight = add_god_node(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )

            global_attr = np.array(
                [
                    float(n_wells),
                    float(np.mean(is_injector)),
                    float(np.mean(depth)),
                    float(np.std(depth)),
                    float(np.sum(inj_rate)),
                    float(np.sum(abs_inj_rate)),
                ],
                dtype=np.float32,
            ).reshape(1, -1)

            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
                global_attr=torch.tensor(global_attr, dtype=torch.float32),
                y=torch.tensor([float(production_total[-1])], dtype=torch.float32),
                case_id=case_id,
            )
            graphs.append(data)
            all_targets.append(float(production_total[-1]))

    if skipped_empty > 0:
        print(f"Skipped empty-well cases: {skipped_empty}")

    return graphs, np.array(all_targets, dtype=np.float32)


def split_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be > 0.")

    all_idx = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(all_idx, test_size=test_fraction, random_state=seed, shuffle=True)
    val_rel = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_rel, random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx


def split_indices_stratified(
    targets: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    n_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be > 0.")

    n_samples = len(targets)
    all_idx = np.arange(n_samples)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(targets, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return split_indices(n_samples, val_fraction, test_fraction, seed)

    strat_labels = np.digitize(targets, edges[1:-1], right=False)

    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=strat_labels,
    )

    val_rel = val_fraction / (1.0 - test_fraction)
    train_val_labels = strat_labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_rel,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )
    return train_idx, val_idx, test_idx


class GNNRegressor(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        global_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
        pooling: str,
        residual: bool,
        edge_dropout: float,
        learning_rate: float,
        weight_decay: float,
        loss: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_cls = activations.get(activation.lower())
        if activation_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")

        valid_pooling = {"mean", "avg", "sum", "concat_sum_mean_avg"}
        if pooling not in valid_pooling:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        if input_dim == hidden_dim:
            self.residual_proj.append(nn.Identity())
        else:
            self.residual_proj.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
            self.residual_proj.append(nn.Identity())

        self.activation = activation_cls()
        self.dropout = nn.Dropout(dropout)

        pool_multiplier = 3 if pooling == "concat_sum_mean_avg" else 1
        pooled_dim = hidden_dim * pool_multiplier
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + global_dim, hidden_dim),
            activation_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "huber":
            self.loss_fn = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        pooling = self.hparams.pooling
        if pooling in {"mean", "avg"}:
            return global_mean_pool(x, batch)
        if pooling == "sum":
            return global_add_pool(x, batch)
        if pooling == "concat_sum_mean_avg":
            pooled_sum = global_add_pool(x, batch)
            pooled_mean = global_mean_pool(x, batch)
            pooled_avg = global_mean_pool(x, batch)
            return torch.cat([pooled_sum, pooled_mean, pooled_avg], dim=1)
        raise ValueError(f"Unsupported pooling: {pooling}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        global_attr: torch.Tensor,
    ) -> torch.Tensor:
        edge_weight_used = edge_weight
        if self.training and self.hparams.edge_dropout > 0.0 and edge_weight.numel() > 0:
            keep_prob = 1.0 - self.hparams.edge_dropout
            mask = torch.rand(edge_weight.shape[0], device=edge_weight.device) < keep_prob
            if torch.any(mask):
                edge_weight_used = edge_weight.clone()
                edge_weight_used[~mask] = 0.0
        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            x = conv(x, edge_index, edge_weight=edge_weight_used)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            if self.hparams.residual:
                x = x + self.residual_proj[idx](x_in)

        graph_emb = self._pool_graph(x, batch)
        full_emb = torch.cat([graph_emb, global_attr], dim=1)
        out = self.head(full_emb).squeeze(-1)
        return out

    def _step(self, batch: Data, stage: str) -> torch.Tensor:
        preds = self(batch.x, batch.edge_index, batch.edge_weight, batch.batch, batch.global_attr)
        target = batch.y.view(-1)
        loss = self.loss_fn(preds, target)
        mae = torch.mean(torch.abs(preds - target))
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_mae_scaled", mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def evaluate_split(
    model: GNNRegressor,
    graphs: list[Data],
    scaler: GraphScaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds_scaled: list[np.ndarray] = []
    actual_scaled: list[np.ndarray] = []
    case_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch, batch.global_attr)
            preds_scaled.append(out.detach().cpu().numpy())
            actual_scaled.append(batch.y.view(-1).detach().cpu().numpy())
            case_ids.extend(batch.case_id)

    y_pred_scaled = np.concatenate(preds_scaled)
    y_true_scaled = np.concatenate(actual_scaled)

    y_pred = scaler.inverse_targets(y_pred_scaled)
    y_true = scaler.inverse_targets(y_true_scaled)
    return y_true, y_pred, case_ids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_error_scatter_plots(output_dir: Path, split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residual = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))

    axes[0].scatter(y_true, y_pred, s=18, alpha=0.7)
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axes[0].set_xlabel("Actual final field_energy_production_total")
    axes[0].set_ylabel("Predicted final field_energy_production_total")
    axes[0].set_title(f"{split_name}: Predicted vs Actual")

    axes[1].scatter(y_true, residual, s=18, alpha=0.7)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xlabel("Actual final field_energy_production_total")
    axes[1].set_ylabel("Prediction error (pred - actual)")
    axes[1].set_title(f"{split_name}: Error Scatter")

    fig.tight_layout()
    out_path = output_dir / f"{split_name}_prediction_error_scatter.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def save_loss_curve_plot(metrics_csv_path: Path, output_path: Path) -> None:
    if not metrics_csv_path.exists():
        print(f"Skipping loss plot; metrics file not found: {metrics_csv_path}")
        return

    train_epochs: list[float] = []
    train_losses: list[float] = []
    val_epochs: list[float] = []
    val_losses: list[float] = []

    with metrics_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_raw = row.get("epoch", "")
            if not epoch_raw:
                continue
            try:
                epoch = float(epoch_raw)
            except ValueError:
                continue

            train_raw = row.get("train_loss_epoch", "")
            if train_raw:
                try:
                    train_epochs.append(epoch)
                    train_losses.append(float(train_raw))
                except ValueError:
                    pass

            val_raw = row.get("val_loss", "")
            if val_raw:
                try:
                    val_epochs.append(epoch)
                    val_losses.append(float(val_raw))
                except ValueError:
                    pass

    if not train_losses and not val_losses:
        print("Skipping loss plot; no usable loss rows found in metrics.csv")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(train_epochs, train_losses, label="train_loss_epoch")
    if val_losses:
        ax.plot(val_epochs, val_losses, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def save_predictions_csv(
    output_path: Path,
    split_name: str,
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(y_pred - y_true)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "case_id", "actual", "predicted", "error", "abs_error"])
        for cid, yt, yp, ae in zip(case_ids, y_true, y_pred, abs_err):
            writer.writerow([split_name, cid, float(yt), float(yp), float(yp - yt), float(ae)])
    print(f"Saved predictions: {output_path}")


def save_extreme_error_plots(
    output_dir: Path,
    split_name: str,
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 20,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(y_pred - y_true)
    k = min(top_k, len(abs_err))
    if k == 0:
        return

    high_idx = np.argsort(abs_err)[-k:][::-1]
    low_idx = np.argsort(abs_err)[:k]

    def _plot(indices: np.ndarray, title_suffix: str, filename: str) -> None:
        labels = [case_ids[i] for i in indices]
        values = abs_err[indices]
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(indices))))
        ypos = np.arange(len(indices))
        ax.barh(ypos, values)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute error")
        ax.set_title(f"{split_name}: {title_suffix}")
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    _plot(high_idx, f"Top {k} highest-error cases", f"{split_name}_highest_error_cases.png")
    _plot(low_idx, f"Top {k} lowest-error cases", f"{split_name}_lowest_error_cases.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an enhanced KNN-graph PyG model for final total energy generation.")
    parser.add_argument("--h5-path", type=Path, default=Path("minimal_compiled_all.h5"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--knn-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="gelu")
    parser.add_argument("--pooling", choices=["mean", "avg", "sum", "concat_sum_mean_avg"], default="mean")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections across GNN layers.")
    parser.add_argument("--edge-dropout", type=float, default=0.05, help="Probability of dropping edge weights during training.")
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--no-log-target", action="store_true")
    parser.add_argument("--no-god-node", action="store_true")
    parser.add_argument("--no-stratified-split", action="store_true", help="Disable quantile-stratified split by target value.")
    parser.add_argument("--grad-clip-val", type=float, default=1.0)
    parser.add_argument("--extreme-k", type=int, default=20)
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.knn_k < 1:
        raise ValueError("--knn-k must be >= 1")

    L.seed_everything(args.seed, workers=True)
    seed_all(args.seed)

    graphs_raw, targets = load_graphs(args.h5_path, knn_k=args.knn_k, use_god_node=not args.no_god_node)
    if not args.no_stratified_split:
        train_idx, val_idx, test_idx = split_indices_stratified(
            targets=targets,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    else:
        train_idx, val_idx, test_idx = split_indices(len(graphs_raw), args.val_fraction, args.test_fraction, args.seed)

    train_graphs_raw = [graphs_raw[i] for i in train_idx]
    val_graphs_raw = [graphs_raw[i] for i in val_idx]
    test_graphs_raw = [graphs_raw[i] for i in test_idx]

    scaler = GraphScaler(
        whiten=not args.no_whiten,
        pca_components=args.pca_components,
        log_target=not args.no_log_target,
    )
    scaler.fit(train_graphs_raw)

    train_graphs = [scaler.transform_graph(g) for g in train_graphs_raw]
    val_graphs = [scaler.transform_graph(g) for g in val_graphs_raw]
    test_graphs = [scaler.transform_graph(g) for g in test_graphs_raw]

    input_dim = train_graphs[0].x.shape[1]
    global_dim = train_graphs[0].global_attr.shape[1]

    print(f"Loaded {len(graphs_raw)} graphs from {args.h5_path}")
    print(f"Target range: min={float(np.min(targets)):.4f}, max={float(np.max(targets)):.4f}")
    print(f"Node feature dimension after scaling/whitening: {input_dim}")
    print(f"Global feature dimension: {global_dim}")
    print(f"Pooling mode: {args.pooling}")
    print(f"Using god node: {not args.no_god_node}")
    print(f"Split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GNNRegressor(
        input_dim=input_dim,
        global_dim=global_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        pooling=args.pooling,
        residual=not args.no_residual,
        edge_dropout=args.edge_dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss=args.loss,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_loss:.4f}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    logger = CSVLogger(save_dir="lightning_logs", name="geothermal_gnn")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip_val,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_cb.best_model_path
    if not best_path:
        raise RuntimeError("No checkpoint was saved.")

    print(f"Best checkpoint: {best_path}")

    best_model = GNNRegressor.load_from_checkpoint(best_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("mps")
    best_model = best_model.to(device)

    split_graphs = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }

    if args.plots_dir is None:
        plots_dir = Path(logger.log_dir) / "plots"
    else:
        plots_dir = args.plots_dir

    print("\nMetrics in original target units:")
    for split_name, graphs in split_graphs.items():
        y_true, y_pred, case_ids = evaluate_split(
            model=best_model,
            graphs=graphs,
            scaler=scaler,
            batch_size=args.batch_size,
            device=device,
        )
        metrics = compute_metrics(y_true, y_pred)
        print(
            f"  {split_name:<5} | MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
        )
        save_error_scatter_plots(plots_dir, split_name, y_true, y_pred)
        save_predictions_csv(plots_dir / f"{split_name}_predictions.csv", split_name, case_ids, y_true, y_pred)
        save_extreme_error_plots(
            output_dir=plots_dir,
            split_name=split_name,
            case_ids=case_ids,
            y_true=y_true,
            y_pred=y_pred,
            top_k=args.extreme_k,
        )

    save_loss_curve_plot(Path(logger.log_dir) / "metrics.csv", plots_dir / "loss_over_time.png")


if __name__ == "__main__":
    main()
