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
from torch_geometric.nn import (
    BatchNorm,
    NNConv,
    HeteroConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    LayerNorm,
)
import torch_geometric.utils as pyg_utils


EDGE_TYPES = [
    ("well", "inj_to_inj", "well"),
    ("well", "ext_to_ext", "well"),
    ("well", "inj_to_ext", "well"),
    ("well", "ext_to_inj", "well"),
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HeteroGraphScaler:
    def __init__(self, whiten: bool, pca_components: int | None) -> None:
        self.node_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.whiten = whiten
        self.pca_components = pca_components
        self.node_pca: PCA | None = None

    def fit(self, graphs: list[HeteroData]) -> None:
        node_train = np.concatenate([g["well"].x.cpu().numpy() for g in graphs], axis=0)
        global_train = np.concatenate(
            [g.global_attr.cpu().numpy() for g in graphs], axis=0
        )

        # Target shape depends on prediction level and whether we filter extractors:
        #   graph-level: (N_graphs, output_dim)
        #   node-level with filter:  (total_ext_nodes, output_dim) — only extractor wells
        #   node-level without filter: (total_nodes, output_dim) — all wells
        prediction_level = getattr(graphs[0], "prediction_level", "graph")
        filter_ext = getattr(graphs[0], "filter_extractors", True)
        if prediction_level == "node" and filter_ext:
            y_raw = np.concatenate(
                [
                    g.y.cpu().numpy()[g["well"].is_injector.cpu().numpy() < 0.5]
                    for g in graphs
                ],
                axis=0,
            )
        elif prediction_level == "node":
            y_raw = np.concatenate([g.y.cpu().numpy() for g in graphs], axis=0)
        else:
            y_raw = np.concatenate([g.y.cpu().numpy() for g in graphs], axis=0)

        node_scaled = self.node_scaler.fit_transform(node_train)
        if self.whiten:
            self.node_pca = PCA(
                n_components=self.pca_components, whiten=True, random_state=42
            )
            self.node_pca.fit(node_scaled)

        self.global_scaler.fit(global_train)

        edge_train_list = []
        for g in graphs:
            for etype in EDGE_TYPES:
                if g[etype].edge_attr.shape[0] > 0:
                    edge_train_list.append(g[etype].edge_attr.cpu().numpy())
        if len(edge_train_list) > 0:
            edge_train = np.concatenate(edge_train_list, axis=0)
            self.edge_scaler.fit(edge_train)

        self.target_scaler.fit(y_raw)

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
            e_attr = graph[edge_type].edge_attr.cpu().numpy()
            if e_attr.shape[0] > 0:
                e_attr_scaled = self.edge_scaler.transform(e_attr)
                transformed[edge_type].edge_attr = torch.tensor(
                    e_attr_scaled, dtype=torch.float32
                )
            else:
                transformed[edge_type].edge_attr = graph[edge_type].edge_attr

        global_np = graph.global_attr.cpu().numpy()
        global_scaled = self.global_scaler.transform(global_np)
        transformed.global_attr = torch.tensor(global_scaled, dtype=torch.float32)

        # Target scaling depends on prediction level
        prediction_level = getattr(graph, "prediction_level", "graph")
        filter_ext = getattr(graph, "filter_extractors", True)
        y_raw = graph.y.cpu().numpy()
        if prediction_level == "node" and filter_ext:
            is_ext = graph["well"].is_injector.cpu().numpy() < 0.5
            y_scaled = np.zeros_like(y_raw, dtype=np.float32)
            if np.any(is_ext):
                y_scaled[is_ext] = self.target_scaler.transform(y_raw[is_ext, :])
        elif prediction_level == "node":
            y_scaled = self.target_scaler.transform(y_raw)
        else:
            y_scaled = self.target_scaler.transform(y_raw)

        transformed.y = torch.tensor(y_scaled, dtype=torch.float32)
        transformed.prediction_level = prediction_level
        transformed.filter_extractors = filter_ext
        transformed.case_id = graph.case_id

        return transformed

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled)


# Feature group column indices for ablation.
# Node features layout: [inj_rate(0), depth(1), perm_x(2), perm_y(3), perm_z(4),
#                        porosity(5), temp0(6), press0(7),
#                        params_scalar(8:34), vertical_profile(34:59)]
# Edge features layout: [plen(0), t_cost(1), m_perm(2), mx_perm(3), hm_perm(4),
#                        m_poro(5), mx_poro(6), hm_poro(7),
#                        delta_t(8), delta_p(9), grad_t(10), grad_p(11),
#                        m_t(12), m_p(13)]
ABLATION_GROUPS = {
    # Node feature groups
    "vertical_profile": {"type": "node", "cols": list(range(34, 59))},
    "params_scalar": {"type": "node", "cols": list(range(8, 34))},
    "base_perm": {"type": "node", "cols": [2, 3, 4]},  # perm_x/y/z
    "base_thermo": {"type": "node", "cols": [6, 7]},  # temp0, press0
    # Edge feature groups
    "edge_perm": {"type": "edge", "cols": [2, 3, 4]},  # min/max/hm perm
    "edge_poro": {"type": "edge", "cols": [5, 6, 7]},  # min/max/hm poro
    "edge_thermo": {
        "type": "edge",
        "cols": [8, 9, 10, 11, 12, 13],
    },  # T/P deltas, grads, mins
    "edge_all": {"type": "edge", "cols": list(range(14))},  # all edge features
}


def apply_ablation(graphs: list, ablate_groups: list[str]) -> None:
    """Zero out specified feature groups in-place for ablation study."""
    for group_name in ablate_groups:
        if group_name not in ABLATION_GROUPS:
            raise ValueError(
                f"Unknown ablation group '{group_name}'. "
                f"Available: {sorted(ABLATION_GROUPS.keys())}"
            )
        spec = ABLATION_GROUPS[group_name]
        cols = spec["cols"]
        for g in graphs:
            if spec["type"] == "node":
                g["well"].x[:, cols] = 0.0
            elif spec["type"] == "edge":
                for etype in EDGE_TYPES:
                    if g[etype].edge_attr.shape[0] > 0:
                        g[etype].edge_attr[:, cols] = 0.0
        print(
            f"  Ablated '{group_name}': zeroed {len(cols)} {spec['type']} feature columns"
        )


# Number of T/P profile statistics: mean, min, max for T and P
TP_PROFILE_STATS = 6


def load_hetero_graphs(
    h5_path: Path, target: str = "graph_energy_total"
) -> tuple[list[HeteroData], np.ndarray]:
    graphs: list[HeteroData] = []
    all_targets: list[float] = []
    skipped_empty = 0

    # Determine prediction level from target choice
    if target in ("node_wept", "node_tp_next"):
        prediction_level = "node"
    else:
        prediction_level = "graph"

    # Determine output dimension
    output_dim = TP_PROFILE_STATS if target == "node_tp_next" else 1

    with h5py.File(h5_path, "r") as handle:
        for case_id in sorted(handle.keys()):
            group = handle[case_id]
            wells = group["wells"][:]

            # Load target based on --target flag
            if target == "node_wept":
                if "well_wept" in group:
                    well_wept = group["well_wept"][:, 0:1]
                else:
                    well_wept = np.zeros((len(wells), 1), dtype=np.float32)
            elif target == "node_tp_next":
                # T/P profile stats: shape [N_wells, N_timesteps, 6]
                if "well_tp_profiles" in group:
                    tp_profiles = group["well_tp_profiles"][:]
                else:
                    tp_profiles = np.zeros(
                        (len(wells), 2, TP_PROFILE_STATS), dtype=np.float32
                    )
                # t=0 stats are already captured by well_vertical_profile;
                # only the final timestep stats are used as the prediction target
                tp_t1 = tp_profiles[:, -2, :]  # (N_wells, 6) — prediction target
            elif target == "graph_energy_total":
                if (
                    "outputs" in group
                    and "field_energy_production_total" in group["outputs"]
                ):
                    fept = group["outputs"]["field_energy_production_total"][:]
                    target_val = float(fept.flat[-1])
                else:
                    target_val = 0.0
            elif target == "graph_energy_rate":
                if (
                    "outputs" in group
                    and "field_energy_production_rate" in group["outputs"]
                ):
                    fepr = group["outputs"]["field_energy_production_rate"][:]
                    target_val = float(fepr.flat[0])
                else:
                    target_val = 0.0
            else:
                raise ValueError(f"Unknown target: {target}")

            params_scalar = group["params_scalar"][:]

            # Vertical profile: 25 features per well (6 props × 4 stats + n_layers)
            if "well_vertical_profile" in group:
                vertical_profile = group["well_vertical_profile"][:].astype(np.float32)
            else:
                vertical_profile = np.zeros((len(wells), 25), dtype=np.float32)

            n_wells = len(wells)
            if n_wells == 0:
                skipped_empty += 1
                continue

            x = wells["x"].astype(np.float32)
            y = wells["y"].astype(np.float32)
            depth = wells["depth"].astype(np.float32)
            inj_rate = wells["inj_rate"].astype(np.float32)
            perm_x = wells["perm_x"].astype(np.float32)
            perm_y = wells["perm_y"].astype(np.float32)
            perm_z = wells["perm_z"].astype(np.float32)
            porosity = wells["porosity"].astype(np.float32)
            temp0 = wells["temp0"].astype(np.float32)
            press0 = wells["press0"].astype(np.float32)

            is_injector = (inj_rate > 0).astype(np.float32)

            # Node features: [inj_rate, depth, perm_x, perm_y, perm_z, porosity, temp0, press0]
            #                 + params_scalar + vertical_profile (25 features)
            base_features = np.stack(
                [inj_rate, depth, perm_x, perm_y, perm_z, porosity, temp0, press0],
                axis=1,
            )
            params_repeated = np.tile(params_scalar, (n_wells, 1))
            node_features = np.concatenate(
                [base_features, params_repeated, vertical_profile], axis=1
            )

            pos_xy = np.stack([x, y], axis=1)

            data = HeteroData()
            data["well"].x = torch.tensor(node_features, dtype=torch.float32)
            data["well"].pos_xy = torch.tensor(pos_xy, dtype=torch.float32)
            data["well"].is_injector = torch.tensor(is_injector, dtype=torch.float32)

            if "inputs" in group and "geology_edge_index" in group["inputs"]:
                geo_idx = group["inputs"]["geology_edge_index"][:]
                geo_attr = group["inputs"]["geology_edge_attr"][:]
            else:
                geo_idx = np.empty((2, 0), dtype=np.int64)
                geo_attr = np.empty((0, 14), dtype=np.float32)

            if geo_attr.shape[0] > 0:
                pass

            # Ensure every node has at least 2 incoming edges from injectors and 2 from extractors
            # (if there are that many injectors/extractors available).
            diff = pos_xy[:, np.newaxis, :] - pos_xy[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dist, np.inf)

            inj_nodes = np.where(is_injector > 0.5)[0]
            ext_nodes = np.where(is_injector < 0.5)[0]

            new_idx_src = []
            new_idx_dst = []
            new_attr = []

            for dst in range(n_wells):
                if geo_idx.shape[1] > 0:
                    incoming_mask = geo_idx[1, :] == dst
                    existing_srcs = geo_idx[0, incoming_mask]
                else:
                    existing_srcs = np.array([], dtype=np.int64)

                # Check injectors
                existing_inj_srcs = np.intersect1d(existing_srcs, inj_nodes)
                missing_inj = 2 - len(existing_inj_srcs)
                if missing_inj > 0:
                    candidate_inj = np.setdiff1d(inj_nodes, existing_inj_srcs)
                    candidate_inj = candidate_inj[candidate_inj != dst]
                    if len(candidate_inj) > 0:
                        cand_dist = dist[candidate_inj, dst]
                        closest_inj = candidate_inj[np.argsort(cand_dist)[:missing_inj]]
                        for src in closest_inj:
                            new_idx_src.append(src)
                            new_idx_dst.append(dst)
                            null_attr = np.zeros(14, dtype=np.float32)
                            null_attr[0] = dist[src, dst]
                            new_attr.append(null_attr)

                # Check extractors
                existing_ext_srcs = np.intersect1d(existing_srcs, ext_nodes)
                missing_ext = 2 - len(existing_ext_srcs)
                if missing_ext > 0:
                    candidate_ext = np.setdiff1d(ext_nodes, existing_ext_srcs)
                    candidate_ext = candidate_ext[candidate_ext != dst]
                    if len(candidate_ext) > 0:
                        cand_dist = dist[candidate_ext, dst]
                        closest_ext = candidate_ext[np.argsort(cand_dist)[:missing_ext]]
                        for src in closest_ext:
                            new_idx_src.append(src)
                            new_idx_dst.append(dst)
                            null_attr = np.zeros(14, dtype=np.float32)
                            null_attr[0] = dist[src, dst]
                            new_attr.append(null_attr)

            if len(new_idx_src) > 0:
                knn_idx = np.array([new_idx_src, new_idx_dst], dtype=np.int64)
                knn_attr = np.array(new_attr, dtype=np.float32)
                if geo_idx.shape[1] > 0:
                    geo_idx = np.concatenate([geo_idx, knn_idx], axis=1)
                    geo_attr = np.concatenate([geo_attr, knn_attr], axis=0)
                else:
                    geo_idx = knn_idx
                    geo_attr = knn_attr

            if geo_idx.shape[1] > 0:
                idx_src = geo_idx[0, :]
                idx_dst = geo_idx[1, :]

                # Boolean masks for each node's injector status
                src_is_inj = is_injector[idx_src] > 0.5
                dst_is_inj = is_injector[idx_dst] > 0.5

                mask_inj_inj = src_is_inj & dst_is_inj
                mask_ext_ext = (~src_is_inj) & (~dst_is_inj)
                mask_inj_ext = src_is_inj & (~dst_is_inj)
                mask_ext_inj = (~src_is_inj) & dst_is_inj

                edge_dict = {
                    ("well", "inj_to_inj", "well"): mask_inj_inj,
                    ("well", "ext_to_ext", "well"): mask_ext_ext,
                    ("well", "inj_to_ext", "well"): mask_inj_ext,
                    ("well", "ext_to_inj", "well"): mask_ext_inj,
                }

                for etype, mask in edge_dict.items():
                    if np.any(mask):
                        e_idx = geo_idx[:, mask]
                        e_attr = geo_attr[mask, :]

                        # Prune to max K=2 incoming connections per target node
                        keep_mask = np.zeros(e_idx.shape[1], dtype=bool)
                        dst_nodes = e_idx[1, :]
                        for d in np.unique(dst_nodes):
                            d_mask = dst_nodes == d
                            d_lengths = e_attr[d_mask, 0]
                            d_indices = np.where(d_mask)[0]
                            # Sort by path length
                            sort_idx = np.argsort(d_lengths)
                            keep_indices = d_indices[sort_idx[:2]]
                            keep_mask[keep_indices] = True

                        data[etype].edge_index = torch.tensor(
                            e_idx[:, keep_mask], dtype=torch.long
                        )
                        data[etype].edge_attr = torch.tensor(
                            e_attr[keep_mask, :], dtype=torch.float32
                        )
                    else:
                        data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
                        data[etype].edge_attr = torch.empty((0, 8), dtype=torch.float32)

            else:
                for etype in EDGE_TYPES:
                    data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
                    data[etype].edge_attr = torch.empty((0, 8), dtype=torch.float32)

            # Global features: just the well count for now, since params_scalar is on the nodes
            global_features = np.array([n_wells], dtype=np.float32)
            data.global_attr = torch.tensor(
                global_features, dtype=torch.float32
            ).unsqueeze(0)

            # Set target on data object
            if target == "node_tp_next":
                data.y = torch.tensor(
                    tp_t1.astype(np.float32), dtype=torch.float32
                )  # (N_wells, 6)
            elif target == "node_wept":
                data.y = torch.tensor(well_wept, dtype=torch.float32)  # (N_wells, 1)
            else:
                data.y = torch.tensor([[target_val]], dtype=torch.float32)  # (1, 1)
            data.prediction_level = prediction_level
            data.filter_extractors = target == "node_wept"
            data.output_dim = output_dim
            data.case_id = case_id

            graphs.append(data)

            # Stratification target
            if target == "node_tp_next":
                # Use mean of all T/P stats at t=1 for stratification
                all_targets.append(float(np.mean(tp_t1)))
            elif target == "node_wept":
                ext_mask = is_injector < 0.5
                ext_vals = well_wept[ext_mask]
                all_targets.append(
                    float(np.mean(ext_vals)) if len(ext_vals) > 0 else 0.0
                )
            else:
                all_targets.append(target_val)

    if skipped_empty > 0:
        print(f"Skipped {skipped_empty} cases with 0 wells.")

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
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=test_fraction, random_state=seed, shuffle=True
        )
        val_rel = val_fraction / (1.0 - test_fraction)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_rel, random_state=seed, shuffle=True
        )
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
        prediction_level: str = "graph",  # "graph" or "node"
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.loss_type = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.prediction_level = prediction_level

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in EDGE_TYPES:
                e_dim = 14
                nn_edge = nn.Sequential(
                    nn.Linear(e_dim, 32),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, in_dim * hidden_dim),
                )
                conv_dict[edge_type] = NNConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    nn=nn_edge,
                    aggr="mean",
                    root_weight=True,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(LayerNorm(hidden_dim))
            if in_dim == hidden_dim:
                self.residual_proj.append(nn.Identity())
            else:
                self.residual_proj.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)

        if prediction_level == "graph":
            # Graph-level: concat(sum, mean, max) pooling + global features → FNN
            pool_dim = 3 * hidden_dim
            head_input_dim = pool_dim + global_dim
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, self.output_dim),
            )
        else:
            # Node-level: FNN per node
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, self.output_dim),
            )

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "huber":
            self.loss_fn = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = {"well": batch["well"].x}
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index for edge_type in EDGE_TYPES
        }
        edge_attr_dict = {
            edge_type: batch[edge_type].edge_attr for edge_type in EDGE_TYPES
        }

        x = x_dict["well"]

        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            out_dict = conv(
                x_dict={"well": x},
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
            x = out_dict["well"]
            x = norm(x, batch["well"].batch)
            x = self.activation(x)
            x = self.dropout_layer(x)
            if self.hparams.residual:
                x = x + self.residual_proj[idx](x_in)

        if self.prediction_level == "graph":
            node_batch = batch["well"].batch
            x_sum = global_add_pool(x, node_batch)
            x_mean = global_mean_pool(x, node_batch)
            x_max = global_max_pool(x, node_batch)
            graph_embed = torch.cat([x_sum, x_mean, x_max], dim=-1)
            global_feat = batch.global_attr
            graph_embed = torch.cat([graph_embed, global_feat], dim=-1)
            return self.head(graph_embed)  # (B, 1)
        else:
            return self.head(x)  # (N_nodes, 1)

    def _step(self, batch: HeteroData, stage: str) -> torch.Tensor:
        preds = self(batch)
        target = batch.y

        filter_ext = getattr(batch, "filter_extractors", True)
        # filter_extractors is batched by PyG as a tensor; extract scalar
        if isinstance(filter_ext, torch.Tensor):
            should_filter = bool(filter_ext[0].item())
        elif isinstance(filter_ext, list):
            should_filter = bool(filter_ext[0])
        else:
            should_filter = bool(filter_ext)

        if self.prediction_level == "node" and should_filter:
            ext_mask = batch["well"].is_injector < 0.5
            preds = preds[ext_mask]
            target = target[ext_mask]

        loss = self.loss_fn(preds, target)
        mae = torch.mean(torch.abs(preds - target))
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"{stage}_mae_scaled",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.num_graphs,
        )
        return loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
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
    is_node_level = model.prediction_level == "node"
    # Check filter_extractors from first graph
    filter_ext = getattr(graphs[0], "filter_extractors", True)
    should_filter = is_node_level and filter_ext

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            if should_filter:
                is_inj = batch["well"].is_injector.detach().cpu().numpy()
                ext_mask = is_inj < 0.5
                pred_scaled_parts.append(pred.detach().cpu().numpy()[ext_mask])
                true_scaled_parts.append(batch.y.detach().cpu().numpy()[ext_mask])
                for i in range(batch.num_graphs):
                    start = batch["well"].ptr[i].item()
                    end = batch["well"].ptr[i + 1].item()
                    n_ext = np.sum(ext_mask[start:end])
                    case_ids.extend([batch.case_id[i]] * int(n_ext))
            elif is_node_level:
                # All nodes (node_tp_next): no filtering
                pred_scaled_parts.append(pred.detach().cpu().numpy())
                true_scaled_parts.append(batch.y.detach().cpu().numpy())
                for i in range(batch.num_graphs):
                    start = batch["well"].ptr[i].item()
                    end = batch["well"].ptr[i + 1].item()
                    n_nodes = end - start
                    case_ids.extend([batch.case_id[i]] * n_nodes)
            else:
                pred_scaled_parts.append(pred.detach().cpu().numpy())
                true_scaled_parts.append(batch.y.detach().cpu().numpy())
                case_ids.extend(list(batch.case_id))

    y_pred_scaled = np.concatenate(pred_scaled_parts)
    y_true_scaled = np.concatenate(true_scaled_parts)

    y_pred = scaler.inverse_targets(y_pred_scaled)
    y_true = scaler.inverse_targets(y_true_scaled)

    return y_true, y_pred, case_ids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Flatten everything so we measure sequence accuracy over all nodes simultaneously
    yt_flat = y_true.flatten()
    yp_flat = y_pred.flatten()

    mae = float(mean_absolute_error(yt_flat, yp_flat))
    rmse = float(np.sqrt(mean_squared_error(yt_flat, yp_flat)))

    # Median Absolute Error is highly resistant to outliers
    medae = float(np.median(np.abs(yt_flat - yp_flat)))

    # Clamped MAPE: Prevent division by near-zero denominators
    epsilon = np.finfo(np.float32).eps
    # Only calculate MAPE on targets > 1% of mean to avoid extreme singularity distortion
    sig_mask = yt_flat > (np.mean(yt_flat) * 0.01)
    if np.any(sig_mask):
        mape_raw = (
            np.abs(
                (yt_flat[sig_mask] - yp_flat[sig_mask])
                / np.maximum(yt_flat[sig_mask], epsilon)
            )
            * 100.0
        )
        # Clamp to 99th percentile to prevent a single extreme outlier from destroying the mean
        mape_clamped = np.clip(mape_raw, a_min=None, a_max=np.percentile(mape_raw, 99))
        mape = float(np.mean(mape_clamped))
    else:
        mape = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "mape": mape,
        "r2": float(r2_score(yt_flat, yp_flat)),
    }


def save_error_scatter_plots(
    output_dir: Path, split_data: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    alphas = {"train": 0.1, "val": 0.3, "test": 0.5}

    for split_name, (y_true, y_pred) in split_data.items():
        yt_flat = y_true.flatten()
        yp_flat = y_pred.flatten()
        residual = yp_flat - yt_flat

        # Calculate robust bounds focusing on the 95% core distribution
        p02_t, p98_t = np.percentile(yt_flat, [2, 98])
        p02_p, p98_p = np.percentile(yp_flat, [2, 98])
        c_min = min(p02_t, p02_p)
        c_max = max(p98_t, p98_p)

        c = colors.get(split_name, "tab:gray")
        a = alphas.get(split_name, 0.3)

        # 1. Clamped Predicted vs Actual
        axes[0].scatter(yt_flat, yp_flat, s=2, alpha=a, color=c, label=split_name)

        # 2. Percentage Error vs Actual
        pct_err = (residual / np.maximum(yt_flat, 1e-6)) * 100.0
        # Clamp pct error to +/- 500% for readability
        pct_err_clamped = np.clip(pct_err, -500, 500)
        axes[1].scatter(
            yt_flat, pct_err_clamped, s=2, alpha=a, color=c, label=split_name
        )

        # 3. Error KDE Distribution
        import scipy.stats as stats

        kde = stats.gaussian_kde(residual)
        # Plot over a robust range around median error
        eval_points = np.linspace(
            np.median(residual) - 3 * np.std(residual),
            np.median(residual) + 3 * np.std(residual),
            100,
        )
        axes[2].plot(
            eval_points, kde(eval_points), color=c, label=split_name, linewidth=2
        )
        axes[2].fill_between(eval_points, 0, kde(eval_points), color=c, alpha=0.1)

    axes[0].plot([c_min, c_max], [c_min, c_max], linestyle="--", color="red")
    axes[0].set_xlim(c_min, c_max)
    axes[0].set_ylim(c_min, c_max)
    axes[0].set_xlabel("Actual Energy Prod. Rate (Core 96%)")
    axes[0].set_ylabel("Predicted Energy Prod. Rate")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    axes[1].axhline(0.0, linestyle="--", color="red")
    axes[1].set_xlabel("Actual Energy Prod. Rate")
    axes[1].set_ylabel("Percentage Error % (Clamped)")
    axes[1].set_title("Relative Error vs Actual")
    axes[1].legend()

    axes[2].axvline(0.0, linestyle="--", color="red")
    axes[2].set_xlabel("Absolute Error (Pred - Actual)")
    axes[2].set_ylabel("Probability Density")
    axes[2].set_title("KDE Error Distribution")
    axes[2].legend()

    fig.tight_layout()
    out_path = output_dir / "combined_prediction_error_dists.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def save_loss_curve_plot(metrics_csv_path: Path, output_path: Path) -> None:
    if not metrics_csv_path.exists():
        return

    hist: dict[str, dict[str, list[float]]] = {
        "loss": {"epochs": [], "train": [], "val": []},
        "mae_scaled": {"epochs": [], "train": [], "val": []},
    }

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

            for metric in ["loss", "mae_scaled"]:
                train_key = (
                    f"train_{metric}_epoch" if metric == "loss" else f"train_{metric}"
                )
                val_key = f"val_{metric}"

                t_val, v_val = row.get(train_key, ""), row.get(val_key, "")
                if t_val or v_val:
                    if epoch not in hist[metric]["epochs"]:
                        hist[metric]["epochs"].append(epoch)
                        hist[metric]["train"].append(float(t_val) if t_val else np.nan)
                        hist[metric]["val"].append(float(v_val) if v_val else np.nan)

    if not hist["loss"]["epochs"]:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["loss", "mae_scaled"]):
        eps = hist[metric]["epochs"]
        t_arr = np.array(hist[metric]["train"])
        v_arr = np.array(hist[metric]["val"])

        # Only plot where values exist
        t_mask = ~np.isnan(t_arr)
        v_mask = ~np.isnan(v_arr)

        if np.any(t_mask):
            t_plot = t_arr[t_mask]
            # Log scale for massive initial losses
            if np.max(t_plot) > 10 * np.min(t_plot) and np.min(t_plot) > 0:
                ax.semilogy(np.array(eps)[t_mask], t_plot, label=f"Train {metric}")
            else:
                ax.plot(np.array(eps)[t_mask], t_plot, label=f"Train {metric}")

        if np.any(v_mask):
            v_plot = v_arr[v_mask]
            if np.max(v_plot) > 10 * np.min(v_plot) and np.min(v_plot) > 0:
                ax.semilogy(np.array(eps)[v_mask], v_plot, label=f"Val {metric}")
            else:
                ax.plot(np.array(eps)[v_mask], v_plot, label=f"Val {metric}")

        ax.set_xlabel("Epoch")
        ax.set_title(f"Curve: {metric.upper()}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved metric tracking curves: {output_path}")


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

    # We now calculate the mean sequence error for each node individually to track highs/lows
    mean_node_err = np.mean(abs_err, axis=1)

    k = min(top_k, len(mean_node_err))
    if k == 0:
        return

    high_idx = np.argsort(mean_node_err)[-k:][::-1]
    low_idx = np.argsort(mean_node_err)[:k]

    def _plot(indices: np.ndarray, title_suffix: str, filename: str) -> None:
        # Avoid duplicate case_ids since one case has many nodes now
        seen_labels = []
        clean_values = []
        for i in indices:
            seen_labels.append(f"{case_ids[i]}_node{i}")
            clean_values.append(mean_node_err[i])

        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(indices))))
        ypos = np.arange(len(indices))
        ax.barh(ypos, clean_values)
        ax.set_yticks(ypos)
        ax.set_yticklabels(seen_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute error")
        ax.set_title(f"{split_name}: {title_suffix}")
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    _plot(
        high_idx,
        f"Top {k} highest-error cases",
        f"{split_name}_highest_error_cases.png",
    )
    _plot(
        low_idx, f"Top {k} lowest-error cases", f"{split_name}_lowest_error_cases.png"
    )


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

    # Compute node-level mean error to find graphs to plot
    mean_node_err = np.mean(abs_err, axis=1)
    if len(mean_node_err) == 0:
        return

    highest_node_idx = int(np.argmax(mean_node_err))
    lowest_node_idx = int(np.argmin(mean_node_err))

    # Trace the node index back to its parent graph index since graphs contain multiple nodes
    highest_case = case_ids[highest_node_idx]
    lowest_case = case_ids[lowest_node_idx]

    highest_idx = 0
    lowest_idx = 0
    for i, g in enumerate(graphs):
        if g.case_id == highest_case:
            highest_idx = i
        if g.case_id == lowest_case:
            lowest_idx = i

    relation_colors = {
        ("well", "inj_to_ext", "well"): "tab:purple",
        ("well", "inj_to_inj", "well"): "tab:orange",
        ("well", "ext_to_ext", "well"): "tab:green",
        ("well", "ext_to_inj", "well"): "tab:brown",
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
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "inj_to_ext", "well")],
                lw=2,
                label="inj_to_ext",
            ),
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "inj_to_inj", "well")],
                lw=2,
                label="inj_to_inj",
            ),
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "ext_to_ext", "well")],
                lw=2,
                label="ext_to_ext",
            ),
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "ext_to_inj", "well")],
                lw=2,
                label="ext_to_inj",
            ),
        ]

        node_handles, node_labels = ax.get_legend_handles_labels()
        ax.legend(handles=node_handles + rel_handles, loc="best", fontsize=8)

        case_id = case_ids[idx]
        ax.set_title(f"{split_name} {label} error graph\n" f"case={case_id}")
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
    # We have shape (N_nodes, 1) for predictions now. We will just dump their flattened
    # absolute error for logging purposes to avoid massive bloated tables.
    mean_node_err = np.mean(np.abs(y_pred - y_true), axis=1)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "case_id", "node_index", "mean_wept_sequence_abs_error"]
        )
        for i, (cid, ae) in enumerate(zip(case_ids, mean_node_err)):
            writer.writerow([split_name, cid, i, float(ae)])
    print(f"Saved predictions: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hetero GNN with typed edge relations and distance edge features."
    )
    parser.add_argument(
        "--h5-path", type=Path, default=Path("data_test/minimal_compiled_optimized.h5")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--early-stop-patience", type=int, default=60)

    import os

    default_workers = min(8, os.cpu_count() or 1) if hasattr(os, "cpu_count") else 4
    parser.add_argument("--num-workers", type=int, default=default_workers)

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--pooling",
        choices=["mean", "avg", "sum", "concat_sum_mean_max"],
        default="mean",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--stratified-split", action="store_true")
    parser.add_argument("--grad-clip-val", type=float, default=1.0)
    parser.add_argument("--extreme-k", type=int, default=20)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument(
        "--ablate",
        nargs="+",
        default=[],
        metavar="GROUP",
        help=(
            "Feature groups to ablate (zero out). Available: "
            + ", ".join(sorted(ABLATION_GROUPS.keys()))
        ),
    )
    parser.add_argument(
        "--target",
        choices=[
            "node_wept",
            "node_tp_next",
            "graph_energy_total",
            "graph_energy_rate",
        ],
        default="graph_energy_total",
        help=(
            "Prediction target: node-level WEPT, node-level next-timestep T/P, "
            "or graph-level energy."
        ),
    )
    args = parser.parse_args()

    prediction_level = (
        "node" if args.target in ("node_wept", "node_tp_next") else "graph"
    )
    output_dim = TP_PROFILE_STATS if args.target == "node_tp_next" else 1

    L.seed_everything(args.seed, workers=True)
    seed_all(args.seed)

    graphs_raw, targets = load_hetero_graphs(args.h5_path, target=args.target)

    print(f"Prediction mode: {args.target} (level={prediction_level})")

    if args.ablate:
        print(f"Ablation study: removing {args.ablate}")
        apply_ablation(graphs_raw, args.ablate)

    if args.stratified_split:
        train_idx, val_idx, test_idx = split_indices_stratified(
            targets=targets,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    else:
        all_idx = np.arange(len(graphs_raw))
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=args.test_fraction, random_state=args.seed, shuffle=True
        )
        val_rel = args.val_fraction / (1.0 - args.test_fraction)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_rel, random_state=args.seed, shuffle=True
        )

    train_graphs_raw = [graphs_raw[i] for i in train_idx]
    val_graphs_raw = [graphs_raw[i] for i in val_idx]
    test_graphs_raw = [graphs_raw[i] for i in test_idx]

    scaler = HeteroGraphScaler(
        whiten=not args.no_whiten,
        pca_components=args.pca_components,
    )
    scaler.fit(train_graphs_raw)

    train_graphs = [scaler.transform_graph(g) for g in train_graphs_raw]
    val_graphs = [scaler.transform_graph(g) for g in val_graphs_raw]
    test_graphs = [scaler.transform_graph(g) for g in test_graphs_raw]

    input_dim = train_graphs[0]["well"].x.shape[1]
    global_dim = train_graphs[0].global_attr.shape[1]

    print(f"Loaded {len(graphs_raw)} hetero graphs from {args.h5_path}")
    print(
        f"Target range: min={float(np.min(targets)):.4f}, max={float(np.max(targets)):.4f}"
    )
    print(f"Node feature dimension after scaling/whitening: {input_dim}")
    print(f"Global feature dimension: {global_dim}")
    print(
        f"Split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}"
    )

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

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
        prediction_level=prediction_level,
        output_dim=output_dim,
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

    plots_dir = (
        Path(logger.log_dir) / "plots" if args.plots_dir is None else args.plots_dir
    )

    print("\nMetrics in original target units:")

    # Accumulate evaluation predictions for combined scatter plotting
    split_eval_data = {}

    for split_name, graphs in split_graphs.items():
        y_true, y_pred, case_ids = evaluate_split(
            best_model, graphs, scaler, args.batch_size, device
        )
        split_eval_data[split_name] = (y_true, y_pred)

        metrics = compute_metrics(y_true, y_pred)
        print(
            f"  {split_name:<5} | MAE={metrics['mae']:>10.1f} | "
            f"MedAE={metrics['medae']:>10.1f} | RMSE={metrics['rmse']:>10.1f} | "
            f"MAPE={metrics['mape']:>5.1f}% | R2={metrics['r2']:>6.4f}"
        )

        # Per-statistic breakdown for multi-output targets
        if output_dim > 1 and y_true.shape[1] == output_dim:
            stat_names = {
                TP_PROFILE_STATS: [
                    "mean_T",
                    "min_T",
                    "max_T",
                    "mean_P",
                    "min_P",
                    "max_P",
                ],
            }
            names = stat_names.get(output_dim, [f"dim_{j}" for j in range(output_dim)])
            for j, name in enumerate(names):
                m_j = compute_metrics(y_true[:, j : j + 1], y_pred[:, j : j + 1])
                print(
                    f"    {name:>8s}: MAE={m_j['mae']:>8.2f} | "
                    f"MAPE={m_j['mape']:>5.1f}% | R2={m_j['r2']:>7.4f}"
                )
        save_predictions_csv(
            plots_dir / f"{split_name}_predictions.csv",
            split_name,
            case_ids,
            y_true,
            y_pred,
        )
        save_extreme_error_plots(
            plots_dir, split_name, case_ids, y_true, y_pred, top_k=args.extreme_k
        )
        save_extreme_error_graph_plots(
            plots_dir, split_name, graphs, case_ids, y_true, y_pred
        )

    # Plot all validation sequences overlaid on exactly the same graph boundaries
    save_error_scatter_plots(plots_dir, split_eval_data)

    save_loss_curve_plot(
        Path(logger.log_dir) / "metrics.csv", plots_dir / "loss_over_time.png"
    )


if __name__ == "__main__":
    main()
