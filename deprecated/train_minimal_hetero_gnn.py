from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from matplotlib.lines import Line2D

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
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATv2Conv, HeteroConv, global_add_pool, global_mean_pool


EDGE_TYPES = [
    ("well", "inject_extract", "well"),
    ("well", "inject_inject", "well"),
    ("well", "extract_extract", "well"),
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HeteroGraphScaler:
    def __init__(self, whiten: bool, pca_components: int | None, log_target: bool) -> None:
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

    def fit(self, graphs: list[HeteroData]) -> None:
        node_train = np.concatenate([g["well"].x.cpu().numpy() for g in graphs], axis=0)
        global_train = np.concatenate([g.global_attr.cpu().numpy() for g in graphs], axis=0)
        y_raw = np.array([float(g.y.item()) for g in graphs], dtype=np.float32).reshape(-1, 1)

        node_scaled = self.node_scaler.fit_transform(node_train)
        if self.whiten:
            self.node_pca = PCA(n_components=self.pca_components, whiten=True, random_state=42)
            self.node_pca.fit(node_scaled)

        self.global_scaler.fit(global_train)
        y_transformed = self._target_forward(y_raw)
        self.target_scaler.fit(y_transformed)

    def transform_graph(self, graph: HeteroData) -> HeteroData:
        transformed = HeteroData()

        x_np = graph["well"].x.cpu().numpy()
        x_scaled = self.node_scaler.transform(x_np)
        if self.node_pca is not None:
            x_scaled = self.node_pca.transform(x_scaled)
        transformed["well"].x = torch.tensor(x_scaled, dtype=torch.float32)
        transformed["well"].pos_xy = graph["well"].pos_xy
        transformed["well"].is_injector = graph["well"].is_injector

        for edge_type in EDGE_TYPES:
            transformed[edge_type].edge_index = graph[edge_type].edge_index
            transformed[edge_type].edge_attr = graph[edge_type].edge_attr

        global_np = graph.global_attr.cpu().numpy()
        global_scaled = self.global_scaler.transform(global_np)
        transformed.global_attr = torch.tensor(global_scaled, dtype=torch.float32)

        y_raw = np.array([[float(graph.y.item())]], dtype=np.float32)
        y_transformed = self._target_forward(y_raw)
        y_scaled = self.target_scaler.transform(y_transformed).reshape(-1)
        transformed.y = torch.tensor(y_scaled, dtype=torch.float32)
        transformed.case_id = graph.case_id

        return transformed

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        y_unscaled = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))
        y_raw = self._target_inverse(y_unscaled)
        return y_raw.reshape(-1)


def build_typed_knn_edges(
    coords: np.ndarray,
    is_injector: np.ndarray,
    k: int,
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    n_nodes = coords.shape[0]
    result = {
        edge_type: (np.empty((2, 0), dtype=np.int64), np.empty((0, 2), dtype=np.float32))
        for edge_type in EDGE_TYPES
    }
    if n_nodes <= 1:
        return result

    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    src_list: list[int] = []
    dst_list: list[int] = []
    d_list: list[float] = []

    injector_idx = np.where(is_injector > 0.5)[0]
    extractor_idx = np.where(is_injector <= 0.5)[0]

    for i in range(n_nodes):
        for candidate_group in (injector_idx, extractor_idx):
            candidates = candidate_group[candidate_group != i]
            if candidates.size == 0:
                continue

            k_eff = min(k, int(candidates.size))
            candidate_dist = distances[i, candidates]

            if k_eff == candidates.size:
                chosen = candidates
            else:
                chosen_local = np.argpartition(candidate_dist, kth=k_eff - 1)[:k_eff]
                chosen = candidates[chosen_local]

            for j in chosen:
                d = float(distances[i, j])
                src_list.append(i)
                dst_list.append(int(j))
                d_list.append(d)

    src_arr = np.array(src_list, dtype=np.int64)
    dst_arr = np.array(dst_list, dtype=np.int64)
    d_arr = np.array(d_list, dtype=np.float32)

    if d_arr.size == 0:
        return result

    sigma = float(np.median(d_arr)) + 1e-6
    d_norm = d_arr / sigma
    d_rbf = np.exp(-d_norm)
    edge_attr_all = np.stack([d_norm, d_rbf], axis=1).astype(np.float32)

    rel_masks = {
        ("well", "inject_inject", "well"): (is_injector[src_arr] > 0.5) & (is_injector[dst_arr] > 0.5),
        ("well", "extract_extract", "well"): (is_injector[src_arr] < 0.5) & (is_injector[dst_arr] < 0.5),
    }
    rel_masks[("well", "inject_extract", "well")] = ~(rel_masks[("well", "inject_inject", "well")] | rel_masks[("well", "extract_extract", "well")])

    for edge_type in EDGE_TYPES:
        mask = rel_masks[edge_type]
        if np.any(mask):
            edge_index = np.vstack([src_arr[mask], dst_arr[mask]]).astype(np.int64)
            edge_attr = edge_attr_all[mask]
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 2), dtype=np.float32)
        result[edge_type] = (edge_index, edge_attr)

    return result


def load_hetero_graphs(h5_path: Path, knn_k: int) -> tuple[list[HeteroData], np.ndarray]:
    graphs: list[HeteroData] = []
    all_targets: list[float] = []
    skipped_empty = 0

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
            node_features = np.stack([x, y, depth, is_injector, inj_rate, abs_inj_rate], axis=1).astype(np.float32)
            typed_edges = build_typed_knn_edges(coords, is_injector, k=knn_k)

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

            data = HeteroData()
            data["well"].x = torch.tensor(node_features, dtype=torch.float32)
            data["well"].pos_xy = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
            data["well"].is_injector = torch.tensor(is_injector, dtype=torch.float32)
            for edge_type, (edge_index, edge_attr) in typed_edges.items():
                data[edge_type].edge_index = torch.tensor(edge_index, dtype=torch.long)
                data[edge_type].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

            data.global_attr = torch.tensor(global_attr, dtype=torch.float32)
            data.y = torch.tensor([float(production_total[-1])], dtype=torch.float32)
            data.case_id = case_id
            graphs.append(data)
            all_targets.append(float(production_total[-1]))

    if skipped_empty > 0:
        print(f"Skipped empty-well cases: {skipped_empty}")

    return graphs, np.array(all_targets, dtype=np.float32)


def split_indices_stratified(
    targets: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    n_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(targets)
    all_idx = np.arange(n_samples)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(targets, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        train_val_idx, test_idx = train_test_split(all_idx, test_size=test_fraction, random_state=seed, shuffle=True)
        val_rel = val_fraction / (1.0 - test_fraction)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_rel, random_state=seed, shuffle=True)
        return train_idx, val_idx, test_idx

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


class HeteroGNNRegressor(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        global_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pooling: str,
        residual: bool,
        learning_rate: float,
        weight_decay: float,
        loss: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        valid_pooling = {"mean", "avg", "sum", "concat_sum_mean_avg"}
        if pooling not in valid_pooling:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            conv_dict = {
                edge_type: GATv2Conv(
                    (in_dim, in_dim),
                    hidden_dim,
                    heads=1,
                    concat=False,
                    edge_dim=2,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for edge_type in EDGE_TYPES
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(BatchNorm(hidden_dim))
            if in_dim == hidden_dim:
                self.residual_proj.append(nn.Identity())
            else:
                self.residual_proj.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        pool_multiplier = 3 if pooling == "concat_sum_mean_avg" else 1
        pooled_dim = hidden_dim * pool_multiplier
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
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
        pooled_sum = global_add_pool(x, batch)
        pooled_mean = global_mean_pool(x, batch)
        pooled_avg = global_mean_pool(x, batch)
        return torch.cat([pooled_sum, pooled_mean, pooled_avg], dim=1)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = {"well": batch["well"].x}
        edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in EDGE_TYPES}
        edge_attr_dict = {edge_type: batch[edge_type].edge_attr for edge_type in EDGE_TYPES}

        x = x_dict["well"]
        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            out_dict = conv({"well": x}, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x = out_dict["well"]
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            if self.hparams.residual:
                x = x + self.residual_proj[idx](x_in)

        batch_index = batch["well"].batch
        graph_emb = self._pool_graph(x, batch_index)
        full_emb = torch.cat([graph_emb, batch.global_attr], dim=1)
        out = self.head(full_emb).squeeze(-1)
        return out

    def _step(self, batch: HeteroData, stage: str) -> torch.Tensor:
        preds = self(batch)
        target = batch.y.view(-1)
        loss = self.loss_fn(preds, target)
        mae = torch.mean(torch.abs(preds - target))
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.log(f"{stage}_mae_scaled", mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch.num_graphs)
        return loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
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
    model: HeteroGNNRegressor,
    graphs: list[HeteroData],
    scaler: HeteroGraphScaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    pred_scaled_parts: list[np.ndarray] = []
    true_scaled_parts: list[np.ndarray] = []
    case_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_scaled_parts.append(pred.detach().cpu().numpy())
            true_scaled_parts.append(batch.y.view(-1).detach().cpu().numpy())
            case_ids.extend(batch.case_id)

    y_pred_scaled = np.concatenate(pred_scaled_parts)
    y_true_scaled = np.concatenate(true_scaled_parts)

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
                train_epochs.append(epoch)
                train_losses.append(float(train_raw))

            val_raw = row.get("val_loss", "")
            if val_raw:
                val_epochs.append(epoch)
                val_losses.append(float(val_raw))

    if not train_losses and not val_losses:
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


def save_extreme_error_plots(
    output_dir: Path,
    split_name: str,
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
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


def save_extreme_error_graph_plots(
    output_dir: Path,
    split_name: str,
    graphs: list[HeteroData],
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(y_pred - y_true)
    if len(abs_err) == 0:
        return

    highest_idx = int(np.argmax(abs_err))
    lowest_idx = int(np.argmin(abs_err))

    relation_colors = {
        ("well", "inject_extract", "well"): "tab:purple",
        ("well", "inject_inject", "well"): "tab:orange",
        ("well", "extract_extract", "well"): "tab:green",
    }

    def _plot_case(idx: int, label: str, filename: str) -> None:
        graph = graphs[idx]
        pos_xy = graph["well"].pos_xy.detach().cpu().numpy()
        is_injector = graph["well"].is_injector.detach().cpu().numpy() > 0.5

        fig, ax = plt.subplots(figsize=(7, 7))

        for edge_type in EDGE_TYPES:
            edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
            color = relation_colors[edge_type]
            if edge_index.shape[1] == 0:
                continue

            drawn = set()
            for src, dst in edge_index.T:
                key = tuple(sorted((int(src), int(dst))))
                if key in drawn:
                    continue
                drawn.add(key)
                x1, y1 = pos_xy[int(src)]
                x2, y2 = pos_xy[int(dst)]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.35, linewidth=1.0)

        ax.scatter(
            pos_xy[is_injector, 0],
            pos_xy[is_injector, 1],
            c="tab:blue",
            s=40,
            label="Injector",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter(
            pos_xy[~is_injector, 0],
            pos_xy[~is_injector, 1],
            c="tab:red",
            s=40,
            label="Extractor",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

        rel_handles = [
            Line2D([0], [0], color=relation_colors[("well", "inject_extract", "well")], lw=2, label="inject_extract"),
            Line2D([0], [0], color=relation_colors[("well", "inject_inject", "well")], lw=2, label="inject_inject"),
            Line2D([0], [0], color=relation_colors[("well", "extract_extract", "well")], lw=2, label="extract_extract"),
        ]

        node_handles, node_labels = ax.get_legend_handles_labels()
        ax.legend(handles=node_handles + rel_handles, loc="best", fontsize=8)

        case_id = case_ids[idx]
        ax.set_title(
            f"{split_name} {label} error graph\n"
            f"case={case_id}, actual={y_true[idx]:.3e}, pred={y_pred[idx]:.3e}, abs_err={abs_err[idx]:.3e}"
        )
        ax.set_xlabel("well x")
        ax.set_ylabel("well y")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        out_path = output_dir / filename
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    _plot_case(highest_idx, "highest", f"{split_name}_highest_error_graph_xy.png")
    _plot_case(lowest_idx, "lowest", f"{split_name}_lowest_error_graph_xy.png")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hetero GNN with typed edge relations and distance edge features.")
    parser.add_argument("--h5-path", type=Path, default=Path("minimal_compiled_all.h5"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--knn-k", type=int, default=4, help="Nearest wells per destination type (injector and extractor) for each source well.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--early-stop-patience", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--pooling", choices=["mean", "avg", "sum", "concat_sum_mean_avg"], default="mean")
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--no-log-target", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--stratified-split", action="store_true")
    parser.add_argument("--grad-clip-val", type=float, default=1.0)
    parser.add_argument("--extreme-k", type=int, default=20)
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.knn_k < 1:
        raise ValueError("--knn-k must be >= 1")

    L.seed_everything(args.seed, workers=True)
    seed_all(args.seed)

    graphs_raw, targets = load_hetero_graphs(args.h5_path, knn_k=args.knn_k)

    if args.stratified_split:
        train_idx, val_idx, test_idx = split_indices_stratified(
            targets=targets,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    else:
        all_idx = np.arange(len(graphs_raw))
        train_val_idx, test_idx = train_test_split(all_idx, test_size=args.test_fraction, random_state=args.seed, shuffle=True)
        val_rel = args.val_fraction / (1.0 - args.test_fraction)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=val_rel, random_state=args.seed, shuffle=True)

    train_graphs_raw = [graphs_raw[i] for i in train_idx]
    val_graphs_raw = [graphs_raw[i] for i in val_idx]
    test_graphs_raw = [graphs_raw[i] for i in test_idx]

    scaler = HeteroGraphScaler(
        whiten=not args.no_whiten,
        pca_components=args.pca_components,
        log_target=not args.no_log_target,
    )
    scaler.fit(train_graphs_raw)

    train_graphs = [scaler.transform_graph(g) for g in train_graphs_raw]
    val_graphs = [scaler.transform_graph(g) for g in val_graphs_raw]
    test_graphs = [scaler.transform_graph(g) for g in test_graphs_raw]

    input_dim = train_graphs[0]["well"].x.shape[1]
    global_dim = train_graphs[0].global_attr.shape[1]

    print(f"Loaded {len(graphs_raw)} hetero graphs from {args.h5_path}")
    print(f"Target range: min={float(np.min(targets)):.4f}, max={float(np.max(targets)):.4f}")
    print(f"Node feature dimension after scaling/whitening: {input_dim}")
    print(f"Global feature dimension: {global_dim}")
    print(f"Split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = HeteroGNNRegressor(
        input_dim=input_dim,
        global_dim=global_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        residual=not args.no_residual,
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
    early_stop_cb = EarlyStopping(monitor="val_loss", mode="min", patience=args.early_stop_patience, min_delta=1e-5)
    logger = CSVLogger(save_dir="lightning_logs", name="geothermal_hetero_gnn")

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

    best_model = HeteroGNNRegressor.load_from_checkpoint(best_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("mps")
    best_model = best_model.to(device)

    split_graphs = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }

    plots_dir = Path(logger.log_dir) / "plots" if args.plots_dir is None else args.plots_dir

    print("\nMetrics in original target units:")
    for split_name, graphs in split_graphs.items():
        y_true, y_pred, case_ids = evaluate_split(best_model, graphs, scaler, args.batch_size, device)
        metrics = compute_metrics(y_true, y_pred)
        print(
            f"  {split_name:<5} | MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
        )
        save_error_scatter_plots(plots_dir, split_name, y_true, y_pred)
        save_predictions_csv(plots_dir / f"{split_name}_predictions.csv", split_name, case_ids, y_true, y_pred)
        save_extreme_error_plots(plots_dir, split_name, case_ids, y_true, y_pred, top_k=args.extreme_k)
        save_extreme_error_graph_plots(plots_dir, split_name, graphs, case_ids, y_true, y_pred)

    save_loss_curve_plot(Path(logger.log_dir) / "metrics.csv", plots_dir / "loss_over_time.png")


if __name__ == "__main__":
    main()
