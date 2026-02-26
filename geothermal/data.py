"""Data loading, scaling, splitting, ablation, and top-k% withholding for geothermal GNN."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

from geothermal.model import EDGE_TYPES, TP_PROFILE_STATS

# --------------- Feature ablation ---------------

# Node features layout (after params_scalar removal):
# [inj_rate(0), depth(1), perm_x(2), perm_y(3), perm_z(4),
#  porosity(5), temp0(6), press0(7), vertical_profile(8:33)]
# Edge features layout: [plen(0), t_cost(1), m_perm(2), mx_perm(3), hm_perm(4),
#                         m_poro(5), mx_poro(6), hm_poro(7),
#                         delta_t(8), delta_p(9), grad_t(10), grad_p(11),
#                         m_t(12), m_p(13)]
ABLATION_GROUPS = {
    # Node feature groups
    "vertical_profile": {"type": "node", "cols": list(range(8, 33))},
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


# --------------- Graph scaler ---------------


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


# --------------- Graph loading ---------------


def load_hetero_graphs(
    h5_path: Path, target: str = "graph_energy_total"
) -> tuple[list[HeteroData], np.ndarray]:
    graphs: list[HeteroData] = []
    all_targets: list[float] = []
    skipped_empty = 0

    if target in ("node_wept", "node_tp_final"):
        prediction_level = "node"
    else:
        prediction_level = "graph"

    output_dim = TP_PROFILE_STATS if target == "node_tp_final" else 1

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
            elif target == "node_tp_final":
                if "well_tp_profiles" in group:
                    tp_profiles = group["well_tp_profiles"][:]
                else:
                    tp_profiles = np.zeros(
                        (len(wells), 2, TP_PROFILE_STATS), dtype=np.float32
                    )
                tp_t1 = tp_profiles[:, -2, :]
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

            # Node features: [inj_rate, depth, perm_x, perm_y, perm_z,
            #                  porosity, temp0, press0, vertical_profile(25)]
            base_features = np.stack(
                [inj_rate, depth, perm_x, perm_y, perm_z, porosity, temp0, press0],
                axis=1,
            )
            node_features = np.concatenate([base_features, vertical_profile], axis=1)

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

            # Ensure every node has at least 2 incoming edges from injectors and 2 from extractors
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

                        keep_mask = np.zeros(e_idx.shape[1], dtype=bool)
                        dst_nodes = e_idx[1, :]
                        for d in np.unique(dst_nodes):
                            d_mask = dst_nodes == d
                            d_lengths = e_attr[d_mask, 0]
                            d_indices = np.where(d_mask)[0]
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

            # Global features: just the well count
            global_features = np.array([n_wells], dtype=np.float32)
            data.global_attr = torch.tensor(
                global_features, dtype=torch.float32
            ).unsqueeze(0)

            # Set target on data object
            if target == "node_tp_final":
                data.y = torch.tensor(tp_t1.astype(np.float32), dtype=torch.float32)
            elif target == "node_wept":
                data.y = torch.tensor(well_wept, dtype=torch.float32)
            else:
                data.y = torch.tensor([[target_val]], dtype=torch.float32)
            data.prediction_level = prediction_level
            data.filter_extractors = target == "node_wept"
            data.output_dim = output_dim
            data.case_id = case_id

            graphs.append(data)

            # Stratification target
            if target == "node_tp_final":
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


# --------------- Splitting ---------------


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


# --------------- Top-k% withholding ---------------


def withhold_top_pct(
    graphs: list[HeteroData],
    targets: np.ndarray,
    pct: float,
    output_dir: Path,
) -> tuple[list[HeteroData], np.ndarray]:
    """Remove the top ``pct``% of datapoints by target value.

    Returns the filtered (graphs, targets) and writes:
      - ``withheld_runs.json`` listing removed case IDs and their target values
      - ``withholding_histogram.png`` showing the distribution with the cutoff
    """
    if pct <= 0.0 or pct >= 100.0:
        return graphs, targets

    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = np.percentile(targets, 100.0 - pct)
    keep_mask = targets < threshold
    withheld_mask = ~keep_mask

    withheld_ids = [graphs[i].case_id for i in np.where(withheld_mask)[0]]
    withheld_vals = targets[withheld_mask].tolist()

    # Save manifest
    manifest = {
        "withhold_pct": pct,
        "threshold": float(threshold),
        "n_withheld": int(np.sum(withheld_mask)),
        "n_kept": int(np.sum(keep_mask)),
        "withheld_runs": [
            {"case_id": cid, "target_value": float(val)}
            for cid, val in zip(withheld_ids, withheld_vals)
        ],
    }
    manifest_path = output_dir / "withheld_runs.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved withholding manifest: {manifest_path}")

    # Plot histogram
    n_bins = min(40, max(5, len(np.unique(targets))))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        targets,
        bins=n_bins,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
        label="All runs",
    )
    n_wh_unique = len(np.unique(targets[withheld_mask]))
    if n_wh_unique > 1:
        n_bins_wh = min(n_bins, n_wh_unique)
        ax.hist(
            targets[withheld_mask],
            bins=n_bins_wh,
            alpha=0.8,
            color="crimson",
            edgecolor="white",
            label=f"Withheld top {pct:.0f}%",
        )
    else:
        # Only one unique value — mark it with vertical lines instead
        for v in targets[withheld_mask]:
            ax.axvline(v, color="crimson", alpha=0.6, linewidth=1.5)
        ax.axvline(
            targets[withheld_mask][0],
            color="crimson",
            alpha=0.8,
            linewidth=1.5,
            label=f"Withheld top {pct:.0f}% ({int(np.sum(withheld_mask))} runs)",
        )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Cutoff = {threshold:.2f}",
    )
    ax.set_xlabel("Field Energy Production Total")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Target Values — Top {pct:.0f}% Withheld")
    ax.legend()
    fig.tight_layout()
    hist_path = output_dir / "withholding_histogram.png"
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    print(f"Saved withholding histogram: {hist_path}")

    # Filter
    kept_graphs = [graphs[i] for i in np.where(keep_mask)[0]]
    kept_targets = targets[keep_mask]

    print(
        f"Withheld {np.sum(withheld_mask)} runs (top {pct:.0f}%), "
        f"keeping {len(kept_graphs)} runs."
    )
    return kept_graphs, kept_targets
