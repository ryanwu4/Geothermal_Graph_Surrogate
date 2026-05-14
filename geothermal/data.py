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
from geothermal.economics import (
    discounted_revenue_from_rates,
    resolve_real_discount_rate_from_attrs,
)


def _compute_discounted_net_revenue_from_group(group: h5py.Group) -> float:
    """Fallback computation for graph_discounted_net_revenue targets.

    Uses production/injection energy rate arrays and dataset attrs written by preprocess_h5.
    """
    if "field_energy_production_rate" not in group or "field_energy_injection_rate" not in group:
        raise KeyError(
            "Missing field_energy_production_rate or field_energy_injection_rate in compiled dataset"
        )

    attrs = group.file.attrs
    if "target_graph_discounted_net_revenue_energy_price_kwh" not in attrs:
        raise KeyError(
            "Missing target_graph_discounted_net_revenue_energy_price_kwh in dataset attrs"
        )
    energy_price_kwh = float(attrs["target_graph_discounted_net_revenue_energy_price_kwh"])
    discount_rate = resolve_real_discount_rate_from_attrs(attrs)

    prod = group["field_energy_production_rate"][:].astype(np.float64).reshape(-1)
    inj = group["field_energy_injection_rate"][:].astype(np.float64).reshape(-1)
    if prod.shape != inj.shape:
        raise ValueError(
            f"Shape mismatch for production/injection rates: {prod.shape} vs {inj.shape}"
        )

    return discounted_revenue_from_rates(prod, inj, energy_price_kwh, discount_rate)


class PhysicsContext:
    """Wraps physics tensors so PyG doesn't try to vertically stack variable-sized Z fields across graphs."""

    def __init__(self, d: dict[str, torch.Tensor], full_shape: tuple[int, int, int]):
        self.d = d
        self.full_shape = full_shape


# --------------- Feature ablation ---------------

# Node features layout (after params_scalar removal):
# [inj_rate(0), depth(1), perm_x(2), perm_y(3), perm_z(4),
#  porosity(5), temp0(6), press0(7), vertical_profile(8:33)]
# Edge features layout: [plen(0), t_cost(1), m_perm(2), mx_perm(3), hm_perm(4),
#                         m_poro(5), mx_poro(6), hm_poro(7),
#                         delta_t(8), delta_p(9), grad_t(10), grad_p(11),
#                         m_t(12), m_p(13)]
ABLATION_GROUPS = {
    "base_perm": {"type": "node", "cols": [2, 3, 4]},
    "base_thermo": {"type": "node", "cols": [6, 7]},
    # Profile-only ablations (cols 8..32). Each group zeros a slice of the 25-d
    # well_vertical_profile to isolate which features carry signal. Profile stat
    # order per property is (mean, min, max, std).
    "remove_perm_profile":   {"type": "node", "cols": list(range(8, 20))},   # PermX/Y/Z stats
    "remove_poro_profile":   {"type": "node", "cols": list(range(20, 24))},  # Porosity stats
    "remove_thermo_profile": {"type": "node", "cols": list(range(24, 32))},  # T0/P0 stats
    # Stat-axis ablations: zero stats that are NOT the named kind, keep only that kind.
    "keep_only_means":   {"type": "node", "cols": [9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]},
    "keep_only_extrema": {"type": "node", "cols": [8, 12, 16, 20, 24, 28, 11, 15, 19, 23, 27, 31]},
    "keep_only_std":     {"type": "node", "cols": [8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]},
    # Whole-profile sweeps
    "keep_only_n_layers": {"type": "node", "cols": list(range(8, 32))},  # zero all 24 stat cols, keep col 32
    "remove_all_profile": {"type": "node", "cols": list(range(8, 33))},  # zero entire profile incl. n_layers
}


def hparams_to_data_kwargs(hparams) -> dict:
    """Extract data-pipeline kwargs from a loaded model's hparams.

    Use this after `HeteroGNNRegressor.load_from_checkpoint(...)` to thread the
    correct `node_encoder` / `enrich_global_attr` settings into
    `build_single_hetero_data` / `load_hetero_graphs`. Falls back to legacy
    defaults if the checkpoint predates these fields.

    Inference path:
        model = HeteroGNNRegressor.load_from_checkpoint(...)
        data_kw = hparams_to_data_kwargs(model.hparams)
        raw_graph = build_single_hetero_data(..., **data_kw)
    """
    # `hparams` may be either a Namespace-like object or a plain dict.
    if isinstance(hparams, dict):
        node_encoder = hparams.get("node_encoder", "profile")
        global_dim = int(hparams.get("global_dim", 1))
    else:
        node_encoder = getattr(hparams, "node_encoder", "profile")
        global_dim = int(getattr(hparams, "global_dim", 1))
    enrich_global_attr = global_dim != 1
    return {
        "node_encoder": node_encoder,
        "enrich_global_attr": enrich_global_attr,
    }


def _search_geology_metadata_files() -> tuple[Path | None, Path | None]:
    """Find ``filenum_to_scenario_mapping.csv`` and ``geologies_full*.json``
    via standard relative paths from this file. Returns (csv_path, config_path)
    where either may be None if not found.

    Search order (first match wins for each file):

    CSV:
      1. ``<surrogate-repo>/filenum_to_scenario_mapping.csv`` — preferred location
         for the AL workflow's local copy.
      2. ``<surrogate-repo>/configs/filenum_to_scenario_mapping.csv``
      3. ``<workspace>/GeologicalSimulationWrapper.jl/filenum_to_scenario_mapping.csv``

    Geology config:
      1. ``<surrogate-repo>/configs/geologies_full*.json``
      2. ``<workspace>/geothermal_active_learning/configs/geologies_full*.json``
    """
    here = Path(__file__).resolve()
    # geothermal/data.py -> Geothermal_Graph_Surrogate/geothermal -> Geothermal_Graph_Surrogate
    surrogate_repo = here.parent.parent
    # ... -> omv_geothermal/
    workspace = surrogate_repo.parent
    csv_candidates = [
        surrogate_repo / "filenum_to_scenario_mapping.csv",
        surrogate_repo / "configs" / "filenum_to_scenario_mapping.csv",
        workspace / "GeologicalSimulationWrapper.jl" / "filenum_to_scenario_mapping.csv",
    ]
    cfg_candidates = [
        surrogate_repo / "configs" / "geologies_full_local.json",
        surrogate_repo / "configs" / "geologies_full.json",
        workspace / "geothermal_active_learning" / "configs" / "geologies_full_local.json",
        workspace / "geothermal_active_learning" / "configs" / "geologies_full.json",
    ]
    csv = next((p for p in csv_candidates if p.exists()), None)
    cfg = next((p for p in cfg_candidates if p.exists()), None)
    return csv, cfg


def resolve_geology_indices(
    case_ids: list[str],
    h5_path: Path | str,
) -> np.ndarray | None:
    """Return a per-case geology_index array, derived from case_id patterns +
    optional metadata files + H5 fingerprints.

    Resolution order per case:
      1. AL pattern (``..._iter\\d+_<scenario>_run<runnum>_iter\\d+$``) →
         geology_index = ``runnum // 10000``.
      2. Bootstrap pattern (``v2.5_NNNN`` / ``v2.4_NNNN``) →
         ``filenum_to_scenario_mapping.csv`` filenum → scenario,
         then ``geologies_full*.json`` scenario → geology_index.
      3. Fingerprint fallback: rounded log10-mean PermZ over valid voxels.
         Buckets that share a fingerprint with at least one AL- or bootstrap-
         labelled case inherit that label.

    Returns ``None`` if any case can't be assigned (caller falls back to
    target-only stratification).
    """
    import re
    import csv as _csv
    import h5py
    AL_RE = re.compile(r".*_iter\d+_\d+_run(\d+)_iter\d+$")
    BOOT_RE = re.compile(r"^v2\.[45]_(\d{4})$")

    # Lazy-load the bootstrap CSV + geology config once if we'll need them.
    filenum_to_scenario: dict[int, int] | None = None
    scenario_to_geo: dict[int, int] | None = None

    def _ensure_bootstrap_tables() -> bool:
        nonlocal filenum_to_scenario, scenario_to_geo
        if filenum_to_scenario is not None and scenario_to_geo is not None:
            return True
        csv_p, cfg_p = _search_geology_metadata_files()
        if csv_p is None or cfg_p is None:
            return False
        try:
            fn_map: dict[int, int] = {}
            with open(csv_p) as f:
                for row in _csv.DictReader(f):
                    fn_map[int(row["Num"])] = int(row["Scenario"])
            with open(cfg_p) as f:
                cfg = json.load(f)
            sc_map: dict[int, int] = {
                int(e["scenario"]): int(e["geology_index"]) for e in cfg["geologies"]
            }
            filenum_to_scenario = fn_map
            scenario_to_geo = sc_map
            return True
        except Exception as e:
            print(f"NOTE: could not load geology metadata files: {e}")
            return False

    geo_by_idx: dict[int, int] = {}
    unresolved: list[int] = []

    for i, cid in enumerate(case_ids):
        m_al = AL_RE.match(cid)
        if m_al is not None:
            geo_by_idx[i] = int(m_al.group(1)) // 10000
            continue
        m_boot = BOOT_RE.match(cid)
        if m_boot is not None and _ensure_bootstrap_tables():
            filenum = int(m_boot.group(1))
            scen = filenum_to_scenario.get(filenum)  # type: ignore[union-attr]
            if scen is not None and scen in scenario_to_geo:  # type: ignore[operator]
                geo_by_idx[i] = scenario_to_geo[scen]  # type: ignore[index]
                continue
        unresolved.append(i)

    # Fingerprint fallback for anything still unresolved.
    if unresolved:
        fingerprints: dict[int, float] = {}
        with h5py.File(str(h5_path), "r") as f:
            def _fp(cid: str) -> float | None:
                try:
                    permz = f[cid]["physics_tensors"]["PermZ"][:]
                    valid = f[cid]["physics_tensors"]["valid_mask"][:] > 0.5
                    vals = permz[valid]
                    return float(np.round(np.mean(vals), 3)) if vals.size else None
                except Exception:
                    return None
            for i in unresolved:
                fp = _fp(case_ids[i])
                if fp is not None:
                    fingerprints[i] = fp
            # Seed the fp->geo table from already-labelled cases (up to 200 for speed).
            fp_to_geo: dict[float, int] = {}
            for i, geo in list(geo_by_idx.items())[: min(200, len(geo_by_idx))]:
                fp = _fp(case_ids[i])
                if fp is not None and fp not in fp_to_geo:
                    fp_to_geo[fp] = geo
        labelled_fps = np.array(list(fp_to_geo.keys())) if fp_to_geo else np.array([])
        for i in unresolved:
            fp = fingerprints.get(i)
            if fp is None:
                return None
            if fp in fp_to_geo:
                geo_by_idx[i] = fp_to_geo[fp]
            elif labelled_fps.size > 0:
                nearest = labelled_fps[int(np.argmin(np.abs(labelled_fps - fp)))]
                geo_by_idx[i] = fp_to_geo[float(nearest)]
            else:
                return None

    return np.array([geo_by_idx[i] for i in range(len(case_ids))], dtype=np.int64)


def peek_data_kwargs_from_checkpoint(ckpt_path: Path | str | None) -> dict:
    """Read data-pipeline kwargs from a checkpoint file without instantiating the model.

    Useful when the inference script needs to build graphs *before* the model
    object is constructed (e.g., when graph construction happens before the
    device is chosen and the model is moved to it).

    Returns the kwargs dict consumed by `build_single_hetero_data` /
    `load_hetero_graphs`. If `ckpt_path` is None or doesn't exist, falls back to
    the current train.py defaults (`node_encoder='cnn'`, `enrich_global_attr=True`).
    """
    if ckpt_path is None or not Path(ckpt_path).exists():
        return {"node_encoder": "cnn", "enrich_global_attr": True}
    import torch
    import pathlib as _pl
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([_pl.PosixPath, _pl.WindowsPath])
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = blob.get("hyper_parameters") or blob.get("hparams") or {}
    return hparams_to_data_kwargs(hparams)


def apply_ablation(graphs: list, ablate_groups: list[str]) -> None:
    """Zero out specified node feature groups in-place for ablation study."""
    for group_name in ablate_groups:
        if group_name not in ABLATION_GROUPS:
            raise ValueError(f"Unknown node ablation group '{group_name}'")
        spec = ABLATION_GROUPS[group_name]
        cols = spec["cols"]
        for g in graphs:
            if spec["type"] == "node":
                g["well"].x[:, cols] = 0.0
        print(
            f"  Ablated '{group_name}': zeroed {len(cols)} {spec['type']} feature columns"
        )


# --------------- Graph scaler ---------------


class HeteroGraphScaler:
    def __init__(self, whiten: bool, pca_components: int | None) -> None:
        self.node_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.whiten = whiten
        self.pca_components = pca_components
        self.node_pca: PCA | None = None

    def fit(self, graphs: list[HeteroData]) -> None:
        node_train = np.concatenate([g["well"].x.cpu().numpy() for g in graphs], axis=0)
        global_train = np.concatenate(
            [g.global_attr.cpu().numpy() for g in graphs], axis=0
        )
        y_raw = np.concatenate([g.y.cpu().numpy() for g in graphs], axis=0)

        node_scaled = self.node_scaler.fit_transform(node_train)
        if self.whiten:
            self.node_pca = PCA(
                n_components=self.pca_components, whiten=True, random_state=42
            )
            self.node_pca.fit(node_scaled)

        self.global_scaler.fit(global_train)
        self.target_scaler.fit(y_raw)

    def transform_graph(self, graph: HeteroData) -> HeteroData:
        transformed = HeteroData()

        x_np = graph["well"].x.cpu().numpy()
        x_scaled = self.node_scaler.transform(x_np)
        if self.node_pca is not None:
            x_scaled = self.node_pca.transform(x_scaled)
        transformed["well"].x = torch.tensor(x_scaled, dtype=torch.float32)
        transformed["well"].pos_xy = graph["well"].pos_xy
        transformed["well"].pos_xyz = graph["well"].pos_xyz
        transformed["well"].is_injector = graph["well"].is_injector
        if hasattr(graph["well"], "perf_range"):
            transformed["well"].perf_range = graph["well"].perf_range

        for edge_type in EDGE_TYPES:
            transformed[edge_type].edge_index = graph[edge_type].edge_index
            # No edge features to preserve here - they're dynamically calculated in the CNN

        global_np = graph.global_attr.cpu().numpy()
        global_scaled = self.global_scaler.transform(global_np)
        transformed.global_attr = torch.tensor(global_scaled, dtype=torch.float32)

        y_raw = graph.y.cpu().numpy()
        y_scaled = self.target_scaler.transform(y_raw)

        transformed.y = torch.tensor(y_scaled, dtype=torch.float32)
        # Propagate prediction-level attributes from the source graph instead of
        # hardcoding "graph" / False. Hardcoding silently broke node-level targets
        # (`node_wept`, `node_tp_final`): the loss-step's injector mask, which
        # consults `batch.filter_extractors`, was always False after scaling.
        transformed.prediction_level = getattr(graph, "prediction_level", "graph")
        transformed.filter_extractors = getattr(graph, "filter_extractors", False)
        if hasattr(graph, "output_dim"):
            transformed.output_dim = graph.output_dim
        transformed.case_id = graph.case_id

        # Bring over the physics wrappers
        transformed.physics_context = graph.physics_context

        return transformed

    def inverse_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(y_scaled)


# --------------- Graph loading ---------------


def build_single_hetero_data(
    wells: np.ndarray,
    physics_dict: dict[str, torch.Tensor],
    full_shape: tuple[int, int, int],
    target: str,
    target_val: float,
    vertical_profile: np.ndarray,
    tp_t1: np.ndarray | None = None,
    well_wept: np.ndarray | None = None,
    case_id: str = "infer",
    node_encoder: str = "profile",
    enrich_global_attr: bool = False,
) -> HeteroData:
    n_wells = len(wells)

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

    # Per-well perforation range (z_top, z_bot inclusive). vertical_profile[:, 24]
    # is n_layers (set by extract_vertical_profiles in compile_minimal_geothermal_h5.py).
    # depth from wells is the BOTTOM Z index of the perforation; perf_top is derived.
    n_layers_i = vertical_profile[:, 24].astype(np.int64)
    depth_idx = wells["depth"].astype(np.int64)
    perf_top_i = np.maximum(0, depth_idx - n_layers_i + 1)
    perf_bot_i = depth_idx
    perf_range = np.stack([perf_top_i, perf_bot_i], axis=1).astype(np.int64)
    n_layers = n_layers_i.astype(np.float32)
    perf_top = perf_top_i.astype(np.float32)

    # Node features:
    #   node_encoder == "profile" -> 8 base + 25 vertical_profile = 33 dims (legacy)
    #   node_encoder == "cnn"     -> 9 base scalars: [inj_rate, perf_top, perm_x, perm_y,
    #                                perm_z, porosity, temp0, press0, n_layers].
    #     `depth` (well bottom Z) is replaced by `perf_top` so depth=0 means well at
    #     reservoir surface; n_layers is appended because the slab CNN cannot recover it
    #     (slab Z axis is resampled to exactly span [perf_top, perf_bot] regardless of length).
    #   node_encoder == "hybrid"  -> 8 base + 25 vertical_profile = 33 dims, same as
    #     'profile' mode. The model also runs the node CNN at forward time and
    #     concatenates its embedding to these features.
    if node_encoder == "cnn":
        node_features = np.stack(
            [inj_rate, perf_top, perm_x, perm_y, perm_z, porosity, temp0, press0, n_layers],
            axis=1,
        )
    else:
        # both "profile" and "hybrid" use the same 33-d feature vector here
        base_features = np.stack(
            [inj_rate, depth, perm_x, perm_y, perm_z, porosity, temp0, press0],
            axis=1,
        )
        node_features = np.concatenate([base_features, vertical_profile], axis=1)

    pos_xy = np.stack([x, y], axis=1)
    pos_xyz = np.stack([x, y, depth], axis=1)

    data = HeteroData()
    data["well"].x = torch.tensor(node_features, dtype=torch.float32)
    data["well"].pos_xy = torch.tensor(pos_xy, dtype=torch.float32)
    data["well"].pos_xyz = torch.tensor(pos_xyz, dtype=torch.float32)
    data["well"].is_injector = torch.tensor(is_injector, dtype=torch.float32)
    data["well"].perf_range = torch.tensor(perf_range, dtype=torch.long)
    data.physics_context = PhysicsContext(physics_dict, full_shape)

    # K-NN topology based on 3D continuous distance
    diff = pos_xyz[:, np.newaxis, :] - pos_xyz[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, np.inf)

    inj_nodes = np.where(is_injector > 0.5)[0]
    ext_nodes = np.where(is_injector < 0.5)[0]

    new_idx_src = []
    new_idx_dst = []

    for dst in range(n_wells):
        # Closest 2 injectors
        if len(inj_nodes) > 0:
            candidate_inj = inj_nodes[inj_nodes != dst]
            if len(candidate_inj) > 0:
                cand_dist = dist[candidate_inj, dst]
                closest_inj = candidate_inj[np.argsort(cand_dist)[:2]]
                for src in closest_inj:
                    new_idx_src.append(src)
                    new_idx_dst.append(dst)

        # Closest 2 producers
        if len(ext_nodes) > 0:
            candidate_ext = ext_nodes[ext_nodes != dst]
            if len(candidate_ext) > 0:
                cand_dist = dist[candidate_ext, dst]
                closest_ext = candidate_ext[np.argsort(cand_dist)[:2]]
                for src in closest_ext:
                    new_idx_src.append(src)
                    new_idx_dst.append(dst)

    idx_src = np.array(new_idx_src, dtype=np.int64)
    idx_dst = np.array(new_idx_dst, dtype=np.int64)

    if len(idx_src) > 0:
        src_is_inj = is_injector[idx_src] > 0.5
        dst_is_inj = is_injector[idx_dst] > 0.5

        edge_dict = {
            ("well", "inj_to_inj", "well"): src_is_inj & dst_is_inj,
            ("well", "ext_to_ext", "well"): (~src_is_inj) & (~dst_is_inj),
            ("well", "inj_to_ext", "well"): src_is_inj & (~dst_is_inj),
            ("well", "ext_to_inj", "well"): (~src_is_inj) & dst_is_inj,
        }

        for etype, mask in edge_dict.items():
            if np.any(mask):
                e_idx = np.stack([idx_src[mask], idx_dst[mask]], axis=0)
                data[etype].edge_index = torch.tensor(e_idx, dtype=torch.long)
            else:
                data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        for etype in EDGE_TYPES:
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long)

    # Global features. By default just well count (1-d, legacy behaviour).
    # If enrich_global_attr is set, additionally compute reservoir-mean physics
    # statistics from the physics tensor — these are constant per geology and
    # give the head a direct geology fingerprint (especially geo 8's tight
    # reservoir signature) without going through the GNN message-passing.
    if enrich_global_attr:
        valid_mask = physics_dict["valid_mask"].numpy() > 0.5
        def _mean_active(key: str) -> float:
            vals = physics_dict[key].numpy()
            v = vals[valid_mask]
            return float(np.mean(v)) if v.size else 0.0
        permx_mean = _mean_active("PermX")
        permz_mean = _mean_active("PermZ")
        anisotropy = permz_mean / max(permx_mean, 1e-6)
        global_vec = [
            float(n_wells),
            _mean_active("PermX"),
            _mean_active("PermY"),
            _mean_active("PermZ"),
            _mean_active("Porosity"),
            _mean_active("Temperature0"),
            _mean_active("Pressure0"),
            anisotropy,
        ]
        data.global_attr = torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0)
    else:
        data.global_attr = torch.tensor([n_wells], dtype=torch.float32).unsqueeze(0)

    # Target mapping
    if target == "node_tp_final":
        data.y = torch.tensor(tp_t1.astype(np.float32), dtype=torch.float32)
    elif target == "node_wept":
        data.y = torch.tensor(well_wept, dtype=torch.float32)
    else:
        data.y = torch.tensor([[target_val]], dtype=torch.float32)

    data.prediction_level = (
        "node" if target in ("node_wept", "node_tp_final") else "graph"
    )
    data.filter_extractors = target == "node_wept"
    data.output_dim = TP_PROFILE_STATS if target == "node_tp_final" else 1
    data.case_id = case_id

    return data


def load_hetero_graphs(
    h5_path: Path,
    target: str = "graph_energy_total",
    max_cases: int | None = None,
    node_encoder: str = "profile",
    enrich_global_attr: bool = False,
) -> tuple[list[HeteroData], np.ndarray]:
    graphs: list[HeteroData] = []
    all_targets: list[float] = []
    skipped_empty = 0

    with h5py.File(h5_path, "r") as handle:
        for case_id in sorted(handle.keys()):
            group = handle[case_id]
            wells = group["wells"][:]

            # Load Target
            if target == "node_wept":
                well_wept = (
                    group["well_wept"][:, 0:1]
                    if "well_wept" in group
                    else np.zeros((len(wells), 1), dtype=np.float32)
                )
                target_val = 0.0
                tp_t1 = None

            elif target == "node_tp_final":
                tp_profiles = (
                    group["well_tp_profiles"][:]
                    if "well_tp_profiles" in group
                    else np.zeros((len(wells), 2, TP_PROFILE_STATS), dtype=np.float32)
                )
                tp_t1 = tp_profiles[:, -2, :]
                target_val = 0.0
                well_wept = None

            elif target == "graph_energy_total":
                target_array = group["field_energy_production_total"][:]
                target_val = float(target_array.flat[-1])
                tp_t1, well_wept = None, None

            elif target == "graph_energy_rate":
                target_array = group["field_energy_production_rate"][:]
                target_val = float(target_array.flat[-1])
                tp_t1, well_wept = None, None

            elif target == "graph_discounted_net_revenue":
                if "field_discounted_net_revenue" in group:
                    target_val = float(group["field_discounted_net_revenue"][()])
                else:
                    target_val = _compute_discounted_net_revenue_from_group(group)
                tp_t1, well_wept = None, None
            else:
                raise ValueError(f"Unknown target: {target}")

            n_wells = len(wells)
            if n_wells == 0:
                skipped_empty += 1
                continue

            # Vertical profile: 25 features per well (6 props × 4 stats + n_layers).
            # Required for node_encoder='cnn' / 'hybrid' because perf_range is
            # derived from `n_layers = vertical_profile[:, 24]`. A missing profile
            # in profile-mode is recoverable (zero stats); in CNN modes it would
            # silently produce an all-zero perforation mask and n_layers=0, so we
            # raise loudly instead.
            if "well_vertical_profile" in group:
                vertical_profile = group["well_vertical_profile"][:].astype(np.float32)
            else:
                if node_encoder in ("cnn", "hybrid"):
                    raise KeyError(
                        f"Case {case_id!r} in {h5_path} is missing `well_vertical_profile`, "
                        f"required by node_encoder={node_encoder!r} (used to derive per-well "
                        "perforation range and n_layers). Re-run compile_minimal_geothermal_h5.py."
                    )
                vertical_profile = np.zeros((len(wells), 25), dtype=np.float32)

            physics_dict = {}
            for k in group["physics_tensors"].keys():
                t = torch.tensor(group["physics_tensors"][k][:], dtype=torch.float32)
                physics_dict[k] = t

            # Full shape represents the active depth, X, Y
            full_shape = (
                physics_dict["PermX"].shape[0],
                physics_dict["PermX"].shape[1],
                physics_dict["PermX"].shape[2],
            )

            data = build_single_hetero_data(
                wells=wells,
                physics_dict=physics_dict,
                full_shape=full_shape,
                target=target,
                target_val=target_val,
                vertical_profile=vertical_profile,
                tp_t1=tp_t1,
                well_wept=well_wept,
                case_id=case_id,
                node_encoder=node_encoder,
                enrich_global_attr=enrich_global_attr,
            )

            graphs.append(data)

            if max_cases is not None and len(graphs) >= max_cases:
                break

            # Stratification target
            if target == "node_tp_final":
                all_targets.append(float(np.mean(tp_t1)))
            elif target == "node_wept":
                inj_rate = wells["inj_rate"].astype(np.float32)
                is_injector = (inj_rate > 0).astype(np.float32)
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
    geology_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split.

    If `geology_indices` is supplied, the stratification label is
    ``geology_index * n_bins + target_bin`` so each (geology, target-decile)
    cell gets proportional train/val/test counts. This keeps geo 8 (the
    OOD outlier with ~3-5 test samples under target-only stratification)
    properly represented in held-out splits.
    """
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

    target_bins = np.digitize(targets, edges[1:-1], right=False)

    def _safe_strat(labels: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Sklearn requires every class >= 2. Merge any class with < 2 members
        into ``fallback`` (broadcast) iteratively until safe; if even fallback
        leaves rare classes, return None to disable stratification."""
        labels = labels.astype(np.int64).copy()
        for _ in range(3):  # up to 3 merge rounds
            unique, counts = np.unique(labels, return_counts=True)
            rare = unique[counts < 2]
            if rare.size == 0:
                return labels
            labels = np.where(np.isin(labels, rare), fallback, labels)
        # Last resort: pure fallback.
        unique, counts = np.unique(fallback, return_counts=True)
        return fallback if counts.min() >= 2 else None  # type: ignore[return-value]

    if geology_indices is not None:
        geology_indices = np.asarray(geology_indices, dtype=np.int64)
        if geology_indices.shape != targets.shape:
            raise ValueError(
                f"geology_indices shape {geology_indices.shape} must match targets {targets.shape}"
            )
        # Prefer geology as the primary stratification key: with 1000 cases / 15
        # geologies each geology has 50-200 cases, easily splittable. The joint
        # (geology, target_bin) product creates rare cells we'd have to merge.
        strat_labels = _safe_strat(geology_indices, fallback=geology_indices)
    else:
        strat_labels = _safe_strat(target_bins, fallback=target_bins)

    if strat_labels is None:
        # Falls back to non-stratified random split.
        train_val_idx, test_idx = train_test_split(
            all_idx, test_size=test_fraction, random_state=seed, shuffle=True
        )
    else:
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=strat_labels,
        )

    val_rel = val_fraction / (1.0 - test_fraction)
    if strat_labels is None:
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_rel, random_state=seed, shuffle=True
        )
    else:
        train_val_labels = strat_labels[train_val_idx]
        # Re-safe in case the inner pool has any newly-rare classes.
        train_val_labels_safe = _safe_strat(train_val_labels, fallback=train_val_labels)
        if train_val_labels_safe is None:
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=val_rel, random_state=seed, shuffle=True
            )
        else:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_rel,
                random_state=seed,
                shuffle=True,
                stratify=train_val_labels_safe,
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
