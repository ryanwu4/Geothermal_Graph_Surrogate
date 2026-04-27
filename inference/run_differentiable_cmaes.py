#!/usr/bin/env python3
"""
Differentiable Inference Wrapper for Well Location Optimization (CMA-ES Batch Geologies)

This script loads a trained HeteroGNNRegressor model and multiple
geothermal case graphs. It generates N initial well coordinate sets via Latin Hypercube
Sampling natively across the continuous space, bounded by a specified edge buffer.

The optimization process evaluates combinations iteratively across compute chunks (size M),
running parallel optimization of the contiguous well coordinates simultaneously
across ALL candidate geologies to find a robust mean discounted revenue yield.

For active-learning handoff, the script periodically logs candidate well arrangements
as JSON snapshots, exports per-snapshot Intersect-compatible Julia well configs,
and renders per-snapshot figures that include surrogate-estimated targets over all
geologies in the optimization batch.
"""

from __future__ import annotations

import argparse
import json
import csv
import time
from pathlib import Path
import sys
import pickle
from datetime import datetime
from typing import Any

# Ensure repo-root modules are importable when running as `python inference/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch_geometric.data import Batch
import optuna
from optuna.samplers import CmaEsSampler
from optuna.visualization.matplotlib import plot_optimization_history

from compile_minimal_geothermal_h5 import (
    extract_well_data,
    build_wells_table,
    extract_vertical_profiles,
)
from preprocess_h5 import get_valid_mask, find_z_cutoff, PROPERTIES, PERM_PROPS
from geothermal.data import HeteroGraphScaler, build_single_hetero_data
from geothermal.model import HeteroGNNRegressor


def _load_scaler(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} scaler not found at {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    tqdm.write(f"Loaded {label} scaler from {path}")
    return scaler


def _load_model(path: Path, device: torch.device, label: str) -> HeteroGNNRegressor:
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found at {path}")
    model = HeteroGNNRegressor.load_from_checkpoint(str(path), map_location=device)
    model = model.to(device)
    model.eval()
    tqdm.write(f"Loaded {label} model checkpoint from {path}")
    return model


def _inverse_targets_1d(scaler, values_scaled_1d: np.ndarray) -> np.ndarray:
    flat = values_scaled_1d.reshape(-1, 1)
    return scaler.inverse_targets(flat).reshape(-1)


def _decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_h5_scalar(value.item())
        if value.size == 1:
            return _decode_h5_scalar(value.reshape(-1)[0])
        return [
            _decode_h5_scalar(v) for v in value.tolist()
        ]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_geology_metadata(src: h5py.File, geology_name: str) -> dict[str, Any]:
    metadata_group = None
    if "Metadata" in src:
        metadata_group = src["Metadata"]
    elif "metadata" in src:
        metadata_group = src["metadata"]

    def _read_key(key: str) -> Any:
        if metadata_group is None or key not in metadata_group:
            return None
        return _decode_h5_scalar(metadata_group[key][()])

    rep_num = _read_key("RepNum")
    scenario_name = _read_key("ScenarioName")
    sample_num = _read_key("SampleNum")
    data_format_version = _read_key("DataFormatVersion")
    model_version = _read_key("ModelVersion")
    geology_config_id = rep_num if rep_num not in [None, ""] else scenario_name

    return {
        "geology_name": geology_name,
        "geology_config_id": geology_config_id,
        "rep_num": rep_num,
        "scenario_name": scenario_name,
        "sample_num": sample_num,
        "data_format_version": data_format_version,
        "model_version": model_version,
    }


def _jl_quote(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _sanitize_id_token(value: Any) -> str:
    text = "unknown" if value is None else str(value)
    out = []
    for c in text:
        if c.isalnum() or c in ["_", "-"]:
            out.append(c)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token if token else "unknown"


def _to_julia_wells_text(
    coords_xyz: np.ndarray,
    is_injector_list: list[bool],
    score: float,
    score_label: str,
    geology_revenue: list[float] | None = None,
    geology_energy: list[float] | None = None,
    geology_file: str | None = None,
    geology_name: str | None = None,
    geology_config_id: Any = None,
    geology_scenario_name: Any = None,
    geology_sample_num: Any = None,
    predicted_discounted_revenue: float | None = None,
    predicted_total_energy: float | None = None,
) -> str:
    lines = []
    lines.append("# Auto-generated by Differentiable Inference Proxy")
    lines.append(f"# {score_label}: {score:.6f}")
    if geology_revenue is not None:
        lines.append(
            "# Per-geology discounted revenue: "
            + ", ".join(f"{v:.6f}" for v in geology_revenue)
        )
    if geology_energy is not None:
        lines.append(
            "# Per-geology total energy: "
            + ", ".join(f"{v:.6f}" for v in geology_energy)
        )
    if geology_file is not None:
        lines.append(f"# Geology file: {geology_file}")
    if geology_name is not None:
        lines.append(f"# Geology name: {geology_name}")
    if geology_config_id is not None:
        lines.append(f"# Geology config ID (Metadata/RepNum): {geology_config_id}")
    if geology_scenario_name is not None:
        lines.append(f"# Geology scenario name (Metadata/ScenarioName): {geology_scenario_name}")
    if geology_sample_num is not None:
        lines.append(f"# Geology sample number (Metadata/SampleNum): {geology_sample_num}")
    if predicted_discounted_revenue is not None:
        lines.append(
            f"# Predicted discounted revenue for this geology: {predicted_discounted_revenue:.6f}"
        )
    if predicted_total_energy is not None:
        lines.append(
            f"# Predicted total energy for this geology: {predicted_total_energy:.6f}"
        )

    if geology_config_id is not None:
        lines.append(f"geology_config_id = {_jl_quote(geology_config_id)}")
    if geology_scenario_name is not None:
        lines.append(f"geology_scenario_name = {_jl_quote(geology_scenario_name)}")
    if geology_file is not None:
        lines.append(f"geology_source_file = {_jl_quote(geology_file)}")
    if geology_name is not None:
        lines.append(f"geology_source_name = {_jl_quote(geology_name)}")

    lines.append("wells = [")
    for w, (x, y, z) in enumerate(coords_xyz):
        # Map continuous [X, Y, Z] to 1-based [I, J, K].
        # This mirrors the base differentiable inference export convention.
        j_idx = int(round(float(x))) + 1
        i_idx = int(round(float(y))) + 1
        k_idx = int(round(float(z))) + 1

        is_inj = is_injector_list[w]
        well_type = '"INJECTOR"' if is_inj else '"PRODUCER"'
        rate = 8000.0 if is_inj else -8000.0
        lines.append(f"    ({i_idx}, {j_idx}, {k_idx}, {well_type}, {rate}),")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def _build_snapshot_payload(
    run_id: int,
    iteration: int,
    is_final_iteration: bool,
    coords_xyz: np.ndarray,
    is_injector_list: list[bool],
    geology_files: list[str],
    geology_names: list[str],
    geology_metadata: list[dict[str, Any]],
    revenue_by_geo: np.ndarray,
    energy_by_geo: np.ndarray,
) -> dict[str, Any]:
    wells_json = []
    for w, (x, y, z) in enumerate(coords_xyz):
        is_inj = is_injector_list[w]
        j_idx = int(round(float(x))) + 1
        i_idx = int(round(float(y))) + 1
        k_idx = int(round(float(z))) + 1
        wells_json.append(
            {
                "well_id": int(w),
                "type": "injector" if is_inj else "producer",
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "i_idx": int(i_idx),
                "j_idx": int(j_idx),
                "k_idx": int(k_idx),
                "rate": float(8000.0 if is_inj else -8000.0),
            }
        )

    geology_predictions = []
    for k, geo in enumerate(geology_files):
        geo_meta = geology_metadata[k] if k < len(geology_metadata) else {}
        geology_predictions.append(
            {
                "geology_index": int(k),
                "geology_file": str(geo),
                "geology_name": geology_names[k],
                "geology_config_id": geo_meta.get("geology_config_id"),
                "geology_rep_num": geo_meta.get("rep_num"),
                "geology_scenario_name": geo_meta.get("scenario_name"),
                "geology_sample_num": geo_meta.get("sample_num"),
                "geology_data_format_version": geo_meta.get("data_format_version"),
                "geology_model_version": geo_meta.get("model_version"),
                "discounted_total_revenue": float(revenue_by_geo[k]),
                "total_energy_production": float(energy_by_geo[k]),
            }
        )

    payload = {
        "snapshot_id": f"run{run_id:04d}_iter{iteration:04d}",
        "run_id": int(run_id),
        "iteration": int(iteration),
        "is_final_iteration": bool(is_final_iteration),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wells": wells_json,
        "predictions_by_geology": geology_predictions,
        "summary": {
            "mean_discounted_total_revenue": float(np.mean(revenue_by_geo)),
            "mean_total_energy_production": float(np.mean(energy_by_geo)),
        },
    }
    return payload


def main() -> None:
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Well Placement Differentiable Optimization (Optuna CMA-ES Batched)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to input JSON configuration file",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.5, help="LR for spatial optimization"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index to use (default: 0). Pass -1 to force CPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Override root output directory for active-learning artifacts.",
    )
    args = parser.parse_args()

    # 1. Load Configurations & Device Setting
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    def _resolve_config_path(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (config_dir / p)

    with open(config_path, "r") as f:
        config = json.load(f)

    revenue_checkpoint_raw = config.get(
        "revenue_checkpoint", config.get("checkpoint", "")
    )
    if not revenue_checkpoint_raw:
        raise ValueError("Missing 'revenue_checkpoint' (or legacy 'checkpoint')")
    revenue_checkpoint_path = _resolve_config_path(str(revenue_checkpoint_raw))

    revenue_scaler_raw = config.get(
        "revenue_scaler_path", config.get("scaler_path", "")
    )
    if not revenue_scaler_raw:
        raise ValueError("Missing 'revenue_scaler_path' (or legacy 'scaler_path')")
    revenue_scaler_path = _resolve_config_path(str(revenue_scaler_raw))

    energy_checkpoint_raw = config.get(
        "energy_checkpoint", config.get("checkpoint", "")
    )
    if not energy_checkpoint_raw:
        raise ValueError("Missing 'energy_checkpoint' (or legacy 'checkpoint')")
    energy_checkpoint_path = _resolve_config_path(str(energy_checkpoint_raw))

    energy_scaler_raw = config.get(
        "energy_scaler_path", config.get("scaler_path", "")
    )
    if not energy_scaler_raw:
        raise ValueError("Missing 'energy_scaler_path' (or legacy 'scaler_path')")
    energy_scaler_path = _resolve_config_path(str(energy_scaler_raw))

    revenue_target = config.get("revenue_target", "graph_discounted_net_revenue")
    energy_target = config.get("energy_target", "graph_energy_total")

    geology_files = config.get("geology_h5_files", [config.get("geology_h5_file")])
    if not geology_files or geology_files[0] is None:
        raise ValueError(
            "Missing 'geology_h5_files' or 'geology_h5_file' in configuration"
        )
    geology_files = [str(_resolve_config_path(str(g))) for g in geology_files]

    norm_config_path = _resolve_config_path(
        config.get("norm_config", "norm_config.json")
    )
    if not norm_config_path.exists():
        raise FileNotFoundError(f"Norm config not found: {norm_config_path}")

    with open(norm_config_path, "r") as f:
        norm_config = json.load(f)

    num_samples_N = config.get("num_samples_N", 30)
    batch_size_M = config.get("batch_size_M", 10)
    edge_buffer = config.get("edge_buffer", 5)
    num_log_iter = int(config.get("num_log_iter", 5))
    if num_log_iter <= 0:
        raise ValueError("'num_log_iter' must be >= 1")
    log_every_n_steps = int(config.get("log_every_n_steps", 0))
    if log_every_n_steps < 0:
        raise ValueError("'log_every_n_steps' must be >= 0")

    optimization_steps = int(config.get("optimization_steps", 50))

    if args.gpu >= 0 and torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {args.gpu} requested but only "
                f"{torch.cuda.device_count()} GPU(s) available "
                f"(indices 0–{torch.cuda.device_count() - 1})."
            )
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    tqdm.write(f"Using device: {device}")

    revenue_scaler = _load_scaler(revenue_scaler_path, "revenue")
    energy_scaler = _load_scaler(energy_scaler_path, "energy")

    revenue_model = _load_model(revenue_checkpoint_path, device, "revenue")
    energy_model = _load_model(energy_checkpoint_path, device, "energy")

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.get("output_root", "inference_outputs/cmaes_active_learning"))
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_root / f"cmaes_{timestamp}"
    snapshots_json_dir = run_output_dir / "snapshots_json"
    snapshots_fig_dir = run_output_dir / "snapshots_figures"
    snapshot_well_configs_dir = run_output_dir / "well_configs"
    summary_plots_dir = run_output_dir / "summary_plots"
    for d in [
        run_output_dir,
        snapshots_json_dir,
        snapshots_fig_dir,
        snapshot_well_configs_dir,
        summary_plots_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Saving active-learning artifacts to {run_output_dir}")

    # 2. Preload Base Geologies to Memory Space
    open_h5_handles = [h5py.File(g, "r") for g in geology_files]
    h5_physics_contexts = []
    geology_metadata = []

    first_perm_x_grid = None
    z_max, nx, ny = 0, 0, 0

    tqdm.write("Pre-loading base geology tensors...")
    for geo_idx, src in enumerate(open_h5_handles):
        geo_name = Path(geology_files[geo_idx]).stem
        geo_meta = _read_geology_metadata(src, geology_name=geo_name)
        geology_metadata.append(geo_meta)

        valid_mask = get_valid_mask(src)
        z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
        valid_mask_cropped = valid_mask[:z_cutoff]

        physics_dict = {}
        for prop in PROPERTIES:
            data = src[f"Input/{prop}"][:z_cutoff].astype(np.float32)
            if prop in PERM_PROPS:
                data = np.log10(np.maximum(data, 1e-15))
            p_min = norm_config[prop]["min"]
            p_max = norm_config[prop]["max"]
            normalized = (
                (data - p_min) / (p_max - p_min)
                if p_max > p_min
                else np.zeros_like(data)
            )
            normalized = np.clip(normalized, 0.0, 1.0)
            normalized[~valid_mask_cropped] = 0.0
            physics_dict[prop] = torch.tensor(normalized, dtype=torch.float32)

        if first_perm_x_grid is None:
            first_perm_x_grid = physics_dict["PermX"].clone().cpu().numpy()
            z_max = z_cutoff
            nx = valid_mask.shape[1]
            ny = valid_mask.shape[2]

        physics_dict["valid_mask"] = torch.tensor(
            valid_mask_cropped, dtype=torch.float32
        )
        h5_physics_contexts.append(
            {
                "physics_dict": physics_dict,
                "z_cutoff": z_cutoff,
                "full_shape": (z_cutoff, valid_mask.shape[1], valid_mask.shape[2]),
            }
        )

    geology_names = [Path(g).stem for g in geology_files]

    # Plot base arrays
    perm_x_grids = [
        p["physics_dict"]["PermX"].cpu().numpy() for p in h5_physics_contexts
    ]
    z_slice = perm_x_grids[0].shape[0] // 2
    backgrounds = [grid[z_slice, :, :].T for grid in perm_x_grids]

    # Utilities for rendering
    MANIM_BG = "#000000"
    MANIM_BLUE = "#58C4DD"
    MANIM_ORANGE = "#FF9000"
    MANIM_WHITE = "#FFFFFF"
    MANIM_GREY = "#888888"

    FONT_SIZE = 18
    TITLE_SIZE = 17
    TICK_SIZE = 16
    LEGEND_SIZE = 16
    CBAR_LABEL_SIZE = 16

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.facecolor": MANIM_BG,
            "axes.facecolor": MANIM_BG,
            "axes.edgecolor": MANIM_WHITE,
            "axes.labelcolor": MANIM_WHITE,
            "xtick.color": MANIM_WHITE,
            "ytick.color": MANIM_WHITE,
            "text.color": MANIM_WHITE,
            "legend.facecolor": "#111111",
            "legend.edgecolor": MANIM_GREY,
        }
    )

    def _style_ax(ax):
        ax.set_facecolor(MANIM_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(MANIM_WHITE)
        ax.tick_params(colors=MANIM_WHITE, labelsize=TICK_SIZE)
        ax.xaxis.label.set_color(MANIM_WHITE)
        ax.yaxis.label.set_color(MANIM_WHITE)
        ax.title.set_color(MANIM_WHITE)

    def _save_snapshot_figure(
        snapshot_payload: dict[str, Any],
        fig_path: Path,
    ) -> None:
        fig, axes = plt.subplots(
            1,
            K,
            figsize=(8 * K, 7),
            facecolor=MANIM_BG,
        )
        if K == 1:
            axes = [axes]

        wells = snapshot_payload["wells"]

        for k in range(K):
            ax_map = axes[k]
            _style_ax(ax_map)
            bg = backgrounds[k]
            im = ax_map.imshow(
                bg,
                origin="lower",
                cmap="viridis",
                alpha=0.55,
                extent=[0, perm_x_grids[k].shape[1], 0, perm_x_grids[k].shape[2]],
            )
            cbar = plt.colorbar(im, ax=ax_map)
            cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
            cbar.set_label(
                "Normalized Log PermX", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE
            )
            cbar.outline.set_edgecolor(MANIM_WHITE)

            for w in wells:
                is_inj = w["type"] == "injector"
                marker = "^" if is_inj else "v"
                color = MANIM_BLUE if is_inj else MANIM_ORANGE
                ax_map.scatter(
                    w["x"],
                    w["y"],
                    marker=marker,
                    color=color,
                    s=140,
                    edgecolors=MANIM_WHITE,
                    linewidths=1.0,
                    alpha=0.9,
                )

            ax_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
            ax_map.set_ylabel(
                "Y Coordinate" if k == 0 else "",
                fontsize=FONT_SIZE,
                color=MANIM_WHITE,
            )
            ax_map.set_title(
                f"{geology_names[k]} (iter={snapshot_payload['iteration']})",
                fontsize=TITLE_SIZE,
                color=MANIM_WHITE,
            )

        fig.suptitle(
            f"Run {snapshot_payload['run_id']} Iter {snapshot_payload['iteration']}\n"
            f"Mean Rev={snapshot_payload['summary']['mean_discounted_total_revenue']:.3f}, "
            f"Mean Energy={snapshot_payload['summary']['mean_total_energy_production']:.3f}",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )

        fig.tight_layout()
        fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
        plt.close(fig)

    snapshot_records: list[dict[str, Any]] = []

    def _log_snapshot(
        run_id: int,
        iteration: int,
        is_final_iteration: bool,
        coords_xyz: np.ndarray,
        revenue_by_geo: np.ndarray,
        energy_by_geo: np.ndarray,
    ) -> None:
        payload = _build_snapshot_payload(
            run_id=run_id,
            iteration=iteration,
            is_final_iteration=is_final_iteration,
            coords_xyz=coords_xyz,
            is_injector_list=is_injector_list,
            geology_files=geology_files,
            geology_names=geology_names,
            geology_metadata=geology_metadata,
            revenue_by_geo=revenue_by_geo,
            energy_by_geo=energy_by_geo,
        )
        snapshot_id = payload["snapshot_id"]
        json_path = snapshots_json_dir / f"{snapshot_id}.json"
        fig_path = snapshots_fig_dir / f"{snapshot_id}.png"
        jl_path = snapshot_well_configs_dir / f"{snapshot_id}.jl"

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        _save_snapshot_figure(payload, fig_path)

        jl_text = _to_julia_wells_text(
            coords_xyz=coords_xyz,
            is_injector_list=is_injector_list,
            score=float(payload["summary"]["mean_discounted_total_revenue"]),
            score_label="Mean Discounted Revenue",
            geology_revenue=[
                p["discounted_total_revenue"] for p in payload["predictions_by_geology"]
            ],
            geology_energy=[
                p["total_energy_production"] for p in payload["predictions_by_geology"]
            ],
        )
        with open(jl_path, "w") as f:
            f.write(jl_text)

        per_geo_jl_entries = []
        for pred in payload["predictions_by_geology"]:
            geo_idx = int(pred["geology_index"])
            geo_name = pred["geology_name"]
            geo_file = pred["geology_file"]
            geo_cfg_id = pred.get("geology_config_id")
            geo_scenario_name = pred.get("geology_scenario_name")
            geo_sample_num = pred.get("geology_sample_num")

            geo_cfg_token = _sanitize_id_token(
                geo_cfg_id if geo_cfg_id is not None else f"geo{geo_idx:03d}"
            )
            per_geo_stem = f"{snapshot_id}__geo{geo_idx:02d}__cfg_{geo_cfg_token}"
            per_geo_jl_path = snapshot_well_configs_dir / f"{per_geo_stem}.jl"

            per_geo_jl_text = _to_julia_wells_text(
                coords_xyz=coords_xyz,
                is_injector_list=is_injector_list,
                score=float(pred["discounted_total_revenue"]),
                score_label="Discounted Revenue (Geology-Specific)",
                geology_file=geo_file,
                geology_name=geo_name,
                geology_config_id=geo_cfg_id,
                geology_scenario_name=geo_scenario_name,
                geology_sample_num=geo_sample_num,
                predicted_discounted_revenue=float(pred["discounted_total_revenue"]),
                predicted_total_energy=float(pred["total_energy_production"]),
            )
            with open(per_geo_jl_path, "w") as f:
                f.write(per_geo_jl_text)

            per_geo_jl_entries.append(
                {
                    "geology_index": geo_idx,
                    "geology_name": geo_name,
                    "geology_file": geo_file,
                    "geology_config_id": geo_cfg_id,
                    "geology_scenario_name": geo_scenario_name,
                    "geology_sample_num": geo_sample_num,
                    "well_config_path": str(per_geo_jl_path),
                }
            )

        snapshot_records.append(
            {
                "snapshot_id": snapshot_id,
                "run_id": int(run_id),
                "iteration": int(iteration),
                "json_path": str(json_path),
                "figure_path": str(fig_path),
                "well_config_path": str(jl_path),
                "well_config_paths_by_geology": per_geo_jl_entries,
                "mean_discounted_total_revenue": float(
                    payload["summary"]["mean_discounted_total_revenue"]
                ),
                "mean_total_energy_production": float(
                    payload["summary"]["mean_total_energy_production"]
                ),
            }
        )

    # 3. Optuna CMA-ES Study Setup
    num_wells = len(config["wells"])
    tqdm.write(
        f"Initializing Optuna CMA-ES for {num_samples_N} trials targeting {num_wells} wells "
        f"(objective: discounted revenue)..."
    )

    sampler = CmaEsSampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    x_min_bound, x_max_bound = edge_buffer, nx - 1 - edge_buffer
    y_min_bound, y_max_bound = edge_buffer, ny - 1 - edge_buffer

    is_injector_list = []
    for w in range(num_wells):
        w_type = config["wells"][w].get("type", "injector").lower()
        is_injector_list.append(w_type == "injector")

    # 4. Process in Chunks explicitly asking Optuna
    all_final_revenues = []  # (N,) length array tracing final mean discounted revenues
    all_final_energies = []  # (N,) length array tracing final mean energies
    all_trajectories = []  # List of arrays [steps+1, W, 3]
    all_revenue_tracks = []  # List of arrays [steps+1, K]

    K = len(geology_files)

    tqdm.write("\n--- Starting Batched Global-Local Optuna Pipeline ---")
    for batch_index, chunk_start in enumerate(
        tqdm(range(0, num_samples_N, batch_size_M), desc="Batches", unit="batch"),
        start=1,
    ):
        batch_start_time = time.time()
        M_actual = min(batch_size_M, num_samples_N - chunk_start)

        chunk_trials = [study.ask() for _ in range(M_actual)]
        chunk_cfgs = []
        for trial in chunk_trials:
            cfg = []
            for w in range(num_wells):
                rand_x = trial.suggest_float(f"well_{w}_x", x_min_bound, x_max_bound)
                rand_y = trial.suggest_float(f"well_{w}_y", y_min_bound, y_max_bound)
                depth = min(config["wells"][w].get("depth", z_max), z_max)
                w_type = config["wells"][w].get("type", "injector").lower()
                cfg.append({"x": rand_x, "y": rand_y, "depth": depth, "type": w_type})
            chunk_cfgs.append(cfg)
        chunk_graphs_revenue = []
        chunk_graphs_energy = []

        tqdm.write(
            f"Loading/Building PyG models for batch {batch_index} "
            f"({M_actual} configurations evaluated structurally over {K} geologies)..."
        )
        # Build batched graphs identically
        for m_idx, w_cfg in enumerate(chunk_cfgs):
            for k_idx, (src, p_ctx) in enumerate(
                zip(open_h5_handles, h5_physics_contexts)
            ):
                is_well = np.zeros((p_ctx["z_cutoff"], nx, ny), dtype=np.int32)
                inj_rate = np.zeros((p_ctx["z_cutoff"], nx, ny), dtype=np.float32)
                temp0_full = src["Input/Temperature0"][:]

                for w in w_cfg:
                    ix, iy = int(round(w["x"])), int(round(w["y"]))
                    ix, iy = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)

                    for z in range(int(w["depth"])):
                        if temp0_full[z, ix, iy] <= -900:
                            is_well[z, ix, iy] = -999
                            inj_rate[z, ix, iy] = -999
                        else:
                            is_well[z, ix, iy] = 1
                            inj_rate[z, ix, iy] = (
                                8000.0 if w["type"] == "injector" else -8000.0
                            )

                (
                    x_idx,
                    y_idx,
                    depth,
                    inj,
                    perm_x,
                    perm_y,
                    perm_z,
                    porosity,
                    temp0,
                    press0,
                    depth_centroid,
                ) = extract_well_data(is_well, inj_rate, src)
                wells = build_wells_table(
                    x_idx,
                    y_idx,
                    depth,
                    inj,
                    perm_x,
                    perm_y,
                    perm_z,
                    porosity,
                    temp0,
                    press0,
                )
                vertical_profiles = extract_vertical_profiles(
                    is_well, x_idx, y_idx, src
                )

                raw_graph_revenue = build_single_hetero_data(
                    wells=wells,
                    physics_dict=p_ctx["physics_dict"],
                    full_shape=p_ctx["full_shape"],
                    target=revenue_target,
                    target_val=0.0,
                    vertical_profile=vertical_profiles,
                    case_id=f"run{chunk_start+m_idx}_geo{k_idx}",
                )
                raw_graph_energy = build_single_hetero_data(
                    wells=wells,
                    physics_dict=p_ctx["physics_dict"],
                    full_shape=p_ctx["full_shape"],
                    target=energy_target,
                    target_val=0.0,
                    vertical_profile=vertical_profiles,
                    case_id=f"run{chunk_start+m_idx}_geo{k_idx}",
                )
                chunk_graphs_revenue.append(
                    revenue_scaler.transform_graph(raw_graph_revenue)
                )
                chunk_graphs_energy.append(energy_scaler.transform_graph(raw_graph_energy))

        batch_revenue = Batch.from_data_list(chunk_graphs_revenue).to(device)
        batch_energy = Batch.from_data_list(chunk_graphs_energy).to(device)

        continuous_starts = []
        for w_cfg in chunk_cfgs:
            c = []
            for w in w_cfg:
                c.append([w["x"], w["y"], float(w["depth"])])
            continuous_starts.append(c)

        base_coords = torch.tensor(
            continuous_starts, dtype=torch.float32, device=device
        )  # (M, W, 3)
        base_coords.requires_grad = True

        optimizer = optim.Adam([base_coords], lr=args.learning_rate)

        chunk_coords_hist = [
            base_coords.clone().detach().cpu().numpy()
        ]  # list of (M, W, 3)
        chunk_revenue_hist = []  # list of (M, K)

        def _should_log_step(iteration: int) -> bool:
            if log_every_n_steps <= 0:
                return iteration == optimization_steps
            if iteration == 0:
                return True
            return (iteration % log_every_n_steps) == 0 or iteration == optimization_steps

        def _predict_scaled(coords_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
            expanded = coords_tensor.unsqueeze(1).repeat(1, K, 1, 1).view(-1, 3)
            batch_revenue["well"].pos_xyz = expanded
            batch_energy["well"].pos_xyz = expanded
            revenue_scaled = revenue_model(batch_revenue).view(M_actual, K)
            energy_scaled = energy_model(batch_energy).view(M_actual, K)
            return (
                revenue_scaled.detach().cpu().numpy(),
                energy_scaled.detach().cpu().numpy(),
            )

        def _log_batch_snapshot(
            iteration: int,
            coords_np: np.ndarray,
            revenue_scaled_np: np.ndarray,
            energy_scaled_np: np.ndarray,
            is_final_iteration: bool,
        ) -> None:
            for m in range(M_actual):
                run_id = chunk_start + m
                if ((run_id + 1) % num_log_iter) != 0:
                    continue
                revenue_by_geo = _inverse_targets_1d(
                    revenue_scaler, revenue_scaled_np[m]
                )
                energy_by_geo = _inverse_targets_1d(
                    energy_scaler, energy_scaled_np[m]
                )
                _log_snapshot(
                    run_id=run_id,
                    iteration=iteration,
                    is_final_iteration=is_final_iteration,
                    coords_xyz=coords_np[m],
                    revenue_by_geo=revenue_by_geo,
                    energy_by_geo=energy_by_geo,
                )

        if _should_log_step(0):
            with torch.no_grad():
                revenue_scaled_np, energy_scaled_np = _predict_scaled(base_coords)
            _log_batch_snapshot(
                iteration=0,
                coords_np=base_coords.detach().cpu().numpy(),
                revenue_scaled_np=revenue_scaled_np,
                energy_scaled_np=energy_scaled_np,
                is_final_iteration=False,
            )

        steps_iter = tqdm(
            range(optimization_steps),
            desc=f"Batch {batch_index} optimize",
            unit="step",
            leave=False,
        )
        for step in steps_iter:
            optimizer.zero_grad()

            expanded_coords = base_coords.unsqueeze(1).repeat(1, K, 1, 1).view(-1, 3)
            batch_revenue["well"].pos_xyz = expanded_coords
            batch_energy["well"].pos_xyz = expanded_coords

            predicted_revenue_scaled = revenue_model(batch_revenue).view(M_actual, K)
            chunk_revenue_hist.append(
                predicted_revenue_scaled.clone().detach().cpu().numpy()
            )

            # Gradients cleanly distribute independently!
            loss = -predicted_revenue_scaled.sum()
            loss.backward()

            gradients = base_coords.grad  # (M, W, 3)
            with torch.no_grad():
                for d, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                    mask_lower = (base_coords[:, :, d] <= 1e-4) & (
                        gradients[:, :, d] > 0
                    )
                    mask_upper = (base_coords[:, :, d] >= max_val - 1e-4) & (
                        gradients[:, :, d] < 0
                    )
                    mask_out = mask_lower | mask_upper
                    if mask_out.any():
                        gradients[..., d][mask_out] = 0.0
                        if (
                            base_coords in optimizer.state
                            and "exp_avg" in optimizer.state[base_coords]
                        ):
                            optimizer.state[base_coords]["exp_avg"][..., d][
                                mask_out
                            ] = 0.0

            optimizer.step()

            with torch.no_grad():
                base_coords[:, :, 0].clamp_(0, nx - 1)
                base_coords[:, :, 1].clamp_(0, ny - 1)
                base_coords[:, :, 2].clamp_(0, z_max - 1)

            chunk_coords_hist.append(base_coords.clone().detach().cpu().numpy())

            iteration = step + 1
            if _should_log_step(iteration):
                with torch.no_grad():
                    revenue_scaled_np, energy_scaled_np = _predict_scaled(base_coords)
                _log_batch_snapshot(
                    iteration=iteration,
                    coords_np=base_coords.detach().cpu().numpy(),
                    revenue_scaled_np=revenue_scaled_np,
                    energy_scaled_np=energy_scaled_np,
                    is_final_iteration=(iteration == optimization_steps),
                )

        # Extract Terminal Output
        with torch.no_grad():
            expanded_coords = base_coords.unsqueeze(1).repeat(1, K, 1, 1).view(-1, 3)
            batch_revenue["well"].pos_xyz = expanded_coords
            final_revenue_scaled = (
                revenue_model(batch_revenue).view(M_actual, K).detach().cpu().numpy()
            )
            chunk_revenue_hist.append(final_revenue_scaled)

            batch_energy["well"].pos_xyz = expanded_coords
            final_energy_scaled = (
                energy_model(batch_energy).view(M_actual, K).detach().cpu().numpy()
            )
            final_energy_unnorm = energy_scaler.inverse_targets(
                final_energy_scaled.reshape(-1, 1)
            ).reshape(-1, K)

            final_coords_np = base_coords.detach().cpu().numpy()

        chunk_coords_np = np.stack(chunk_coords_hist, axis=0)  # (steps+1, M, W, 3)
        chunk_coords_np = np.transpose(
            chunk_coords_np, (1, 0, 2, 3)
        )  # (M, steps+1, W, 3)

        chunk_revenue_np = np.stack(chunk_revenue_hist, axis=0)  # (steps+1, M, K)
        chunk_revenue_np = np.transpose(
            chunk_revenue_np, (1, 0, 2)
        )  # (M, steps+1, K)

        for m in range(M_actual):
            all_trajectories.append(chunk_coords_np[m])
            revenue_unnorm = revenue_scaler.inverse_targets(
                chunk_revenue_np[m].reshape(-1, 1)
            ).reshape(-1, K)

            all_revenue_tracks.append(revenue_unnorm)

            final_mean_revenue = revenue_unnorm[-1].mean()

            all_final_revenues.append(final_mean_revenue)
            all_final_energies.append(final_energy_unnorm[m].mean())

            # Baldwinian Step: Give Optuna the local optimized discounted revenue.
            study.tell(chunk_trials[m], float(final_mean_revenue))

        batch_elapsed = time.time() - batch_start_time
        tqdm.write(
            f"Computed batch {batch_index} / {int(np.ceil(num_samples_N / batch_size_M))} in {batch_elapsed:.2f}s"
        )

    tqdm.write("\n--- Optimization Complete ---")

    # 5. Output Identifications
    all_final_revenues = np.array(all_final_revenues)
    all_final_energies = np.array(all_final_energies)
    best_idx = np.argmax(all_final_revenues)
    worst_idx = np.argmin(all_final_revenues)
    sorted_idx = np.argsort(all_final_revenues)
    avg_idx = sorted_idx[len(all_final_revenues) // 2]

    tqdm.write(
        f"Top    Run {best_idx} Discounted Revenue: {all_final_revenues[best_idx]:.3f} | "
        f"Energy: {all_final_energies[best_idx]:.3f}"
    )
    tqdm.write(
        f"Mid    Run {avg_idx} Discounted Revenue: {all_final_revenues[avg_idx]:.3f} | "
        f"Energy: {all_final_energies[avg_idx]:.3f}"
    )
    tqdm.write(
        f"Worst  Run {worst_idx} Discounted Revenue: {all_final_revenues[worst_idx]:.3f} | "
        f"Energy: {all_final_energies[worst_idx]:.3f}"
    )

    # ========================================================
    # FIGURE 1: ALL RUNS COMBINED
    # ========================================================
    tqdm.write("\n--- Generating Figure 1 (All Runs Map) ---")
    fig1, axes1 = plt.subplots(
        1,
        K + 1,
        figsize=(10 * (K + 1), 8),
        gridspec_kw={"width_ratios": [1.2] * K + [1.5]},
        facecolor=MANIM_BG,
    )
    ax_energy1 = axes1[-1]
    _style_ax(ax_energy1)

    for k in range(K):
        ax_map = axes1[k]
        _style_ax(ax_map)
        bg = backgrounds[k]
        im = ax_map.imshow(
            bg,
            origin="lower",
            cmap="viridis",
            alpha=0.55,
            extent=[0, perm_x_grids[k].shape[1], 0, perm_x_grids[k].shape[2]],
        )
        cbar = plt.colorbar(im, ax=ax_map)
        cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
        cbar.set_label(
            "Normalized Log PermX", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE
        )
        cbar.outline.set_edgecolor(MANIM_WHITE)

        for n_idx in range(num_samples_N):
            history = all_trajectories[n_idx]
            for w in range(num_wells):
                is_inj = is_injector_list[w]
                color = MANIM_BLUE if is_inj else MANIM_ORANGE
                marker = "^" if is_inj else "v"

                ax_map.scatter(
                    history[-1, w, 0],
                    history[-1, w, 1],
                    color=color,
                    marker=marker,
                    s=50,
                    alpha=0.6,
                )

        ax_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
        ax_map.set_ylabel(
            "Y Coordinate" if k == 0 else "", fontsize=FONT_SIZE, color=MANIM_WHITE
        )
        geo_name = Path(geology_files[k]).stem
        ax_map.set_title(
            f"{geo_name}\nFinal Positions Only (Z={z_slice})",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )

    steps_range = range(optimization_steps + 1)
    for n_idx in range(num_samples_N):
        mean_track = all_revenue_tracks[n_idx].mean(axis=1)
        ax_energy1.plot(
            steps_range,
            mean_track,
            linestyle="-",
            alpha=0.25,
            linewidth=2.0,
            color=MANIM_BLUE,
        )

    ax_energy1.plot(
        [],
        [],
        linestyle="-",
        alpha=0.5,
        linewidth=2.0,
        color=MANIM_BLUE,
        label="Mean Batch Discounted Revenue",
    )
    ax_energy1.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy1.set_ylabel(
        "Discounted Revenue (Non-Norm)", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy1.set_title(
        f"Discounted Revenue Traces across {num_samples_N} CMA-ES Samples",
        fontsize=TITLE_SIZE,
        color=MANIM_WHITE,
    )
    ax_energy1.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)
    ax_energy1.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    plt.tight_layout()
    plot_path1 = summary_plots_dir / f"well_trajectories_cmaes_{timestamp}_all.png"
    fig1.savefig(plot_path1, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig1)

    # ========================================================
    # FIGURE 2: BEST, WORST, AVERAGE HIGHLIGHTS
    # ========================================================
    tqdm.write("--- Generating Figure 2 (Selections Highlights Map) ---")
    fig2, axes2 = plt.subplots(
        1,
        K + 1,
        figsize=(10 * (K + 1), 8),
        gridspec_kw={"width_ratios": [1.2] * K + [1.5]},
        facecolor=MANIM_BG,
    )
    ax_energy2 = axes2[-1]
    _style_ax(ax_energy2)

    for k in range(K):
        ax_map = axes2[k]
        _style_ax(ax_map)
        bg = backgrounds[k]
        im = ax_map.imshow(
            bg,
            origin="lower",
            cmap="viridis",
            alpha=0.55,
            extent=[0, perm_x_grids[k].shape[1], 0, perm_x_grids[k].shape[2]],
        )
        cbar = plt.colorbar(im, ax=ax_map)
        cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
        cbar.set_label(
            "Normalized Log PermX", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE
        )
        cbar.outline.set_edgecolor(MANIM_WHITE)

        labels_done = False
        colors = {
            "Best": "#00FF00",
            "Avg": "#58C4DD",
            "Worst": "#FF0000",
        }  # Green, Blue, Red
        linewidths = {"Best": 4.0, "Avg": 2.5, "Worst": 2.5}
        subsets = [("Best", best_idx), ("Avg", avg_idx), ("Worst", worst_idx)]

        for trace_name, idx in subsets:
            history = all_trajectories[idx]
            trace_color = colors[trace_name]
            lw = linewidths[trace_name]

            for w in range(num_wells):
                is_inj = is_injector_list[w]
                marker_end = "^" if is_inj else "v"

                ax_map.plot(
                    history[:, w, 0],
                    history[:, w, 1],
                    color=trace_color,
                    alpha=0.9,
                    linestyle="-",
                    linewidth=lw,
                    label=(
                        trace_name if (w == 0 and not labels_done and k == 0) else None
                    ),
                )

                # Starting circle
                ax_map.scatter(
                    history[0, w, 0],
                    history[0, w, 1],
                    color=trace_color,
                    marker="o",
                    s=80,
                    edgecolors=MANIM_WHITE,
                    linewidths=1.2,
                    zorder=5,
                )

                # Ending marker
                ax_map.scatter(
                    history[-1, w, 0],
                    history[-1, w, 1],
                    color=trace_color,
                    marker=marker_end,
                    s=200,
                    edgecolors=MANIM_WHITE,
                    linewidths=1.2,
                    zorder=6 + (2 if trace_name == "Best" else 0),
                )
            labels_done = True

        ax_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
        ax_map.set_ylabel(
            "Y Coordinate" if k == 0 else "", fontsize=FONT_SIZE, color=MANIM_WHITE
        )
        geo_name = Path(geology_files[k]).stem
        ax_map.set_title(
            f"{geo_name}\nBest/Avg/Worst PermX (Z={z_slice})",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )
        if k == 0:
            ax_map.legend(
                fontsize=LEGEND_SIZE,
                facecolor="#111111",
                edgecolor=MANIM_GREY,
                labelcolor=MANIM_WHITE,
            )

    # Energy Chart for Highlights
    subsets = [("Best", best_idx), ("Avg", avg_idx), ("Worst", worst_idx)]
    colors = {
        "Best": "#00FF00",
        "Avg": "#58C4DD",
        "Worst": "#FF0000",
    }  # Green, Blue, Red
    linewidths = {"Best": 4.0, "Avg": 2.5, "Worst": 2.5}
    for trace_name, idx in subsets:
        mean_track = all_revenue_tracks[idx].mean(axis=1)  # trace mean across geologies
        ax_energy2.plot(
            steps_range,
            mean_track,
            linestyle="-",
            alpha=0.9,
            linewidth=linewidths[trace_name],
            color=colors[trace_name],
            label=trace_name,
        )

    ax_energy2.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy2.set_ylabel(
        "Discounted Revenue (Non-Norm)", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy2.set_title(
        "Discounted Revenue Traces (Highlights)", fontsize=TITLE_SIZE, color=MANIM_WHITE
    )
    ax_energy2.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)
    ax_energy2.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    plt.tight_layout()
    plot_path2 = summary_plots_dir / f"well_trajectories_cmaes_{timestamp}_highlights.png"
    fig2.savefig(plot_path2, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig2)

    # ========================================================
    # FIGURE 3: HEATMAP OVERALL DISTRIBUTION
    # ========================================================
    tqdm.write("--- Generating Figure 3 (Optimization Layout Heatmaps) ---")
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7), facecolor=MANIM_BG)
    _style_ax(axes3[0])
    _style_ax(axes3[1])

    final_inj_x, final_inj_y = [], []
    final_prd_x, final_prd_y = [], []

    for n_idx in range(num_samples_N):
        history = all_trajectories[n_idx]
        for w in range(num_wells):
            fx, fy = history[-1, w, 0], history[-1, w, 1]
            if is_injector_list[w]:
                final_inj_x.append(fx)
                final_inj_y.append(fy)
            else:
                final_prd_x.append(fx)
                final_prd_y.append(fy)

    # Injector Heatmap
    hb_inj = axes3[0].hexbin(
        final_inj_x,
        final_inj_y,
        gridsize=20,
        cmap="Blues",
        extent=[0, nx, 0, ny],
        mincnt=1,
        alpha=0.95,
        edgecolor="none",
    )
    cb_inj = plt.colorbar(hb_inj, ax=axes3[0])
    cb_inj.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cb_inj.set_label("Count", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE)
    cb_inj.outline.set_edgecolor(MANIM_WHITE)
    axes3[0].set_title(
        f"Injector Final Placements", fontsize=TITLE_SIZE, color=MANIM_WHITE
    )

    # Producer Heatmap
    hb_prd = axes3[1].hexbin(
        final_prd_x,
        final_prd_y,
        gridsize=20,
        cmap="Oranges",
        extent=[0, nx, 0, ny],
        mincnt=1,
        alpha=0.95,
        edgecolor="none",
    )
    cb_prd = plt.colorbar(hb_prd, ax=axes3[1])
    cb_prd.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cb_prd.set_label("Count", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE)
    cb_prd.outline.set_edgecolor(MANIM_WHITE)
    axes3[1].set_title(
        f"Producer Final Placements", fontsize=TITLE_SIZE, color=MANIM_WHITE
    )

    for ax in axes3:
        ax.set_xlabel("X Coordinate", color=MANIM_WHITE, fontsize=FONT_SIZE)
        ax.set_ylabel("Y Coordinate", color=MANIM_WHITE, fontsize=FONT_SIZE)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)

    plt.tight_layout()
    plot_path3 = summary_plots_dir / f"well_trajectories_cmaes_{timestamp}_heatmap.png"
    fig3.savefig(plot_path3, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig3)

    tqdm.write(f"Saved CMA-ES All Runs plot to {plot_path1}")
    tqdm.write(f"Saved CMA-ES Highlights plot to {plot_path2}")
    tqdm.write(f"Saved CMA-ES Heatmap distribution to {plot_path3}")

    # ========================================================
    # FIGURE 4: OPTUNA OPTIMIZATION HISTORY
    # ========================================================
    tqdm.write("--- Generating Figure 4 (Optuna Convergence History) ---")
    ax_hist = plot_optimization_history(study)
    fig4 = ax_hist.figure
    fig4.set_size_inches(12, 7)
    fig4.set_facecolor(MANIM_BG)
    _style_ax(ax_hist)
    
    # Style the legend for dark mode since optuna creates its own
    legend = ax_hist.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#111111")
        legend.get_frame().set_edgecolor(MANIM_GREY)
        for text in legend.get_texts():
            text.set_color(MANIM_WHITE)
            
    plot_path4 = summary_plots_dir / f"well_trajectories_cmaes_{timestamp}_history.png"
    fig4.tight_layout()
    fig4.savefig(plot_path4, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig4)
    tqdm.write(f"Saved CMA-ES Optuna History to {plot_path4}")

    csv_path = run_output_dir / f"well_trajectories_cmaes_{timestamp}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "well_id",
                "well_type",
                "depth_z",
                "final_x",
                "final_y",
                "mean_discounted_total_revenue",
                "mean_total_energy",
            ]
        )
        for n_idx in range(num_samples_N):
            history = all_trajectories[n_idx]
            mean_revenue = all_final_revenues[n_idx]
            mean_energy = all_final_energies[n_idx]
            final_coords = history[-1]
            for w in range(num_wells):
                x, y, z = final_coords[w]
                w_type = "injector" if is_injector_list[w] else "producer"
                writer.writerow([n_idx, w, w_type, z, x, y, mean_revenue, mean_energy])
    tqdm.write(f"Exported complete final configuration tables to {csv_path}")

    snapshot_manifest_path = run_output_dir / f"snapshot_manifest_{timestamp}.json"
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "objective": "maximize_discounted_total_revenue",
        "num_samples_N": int(num_samples_N),
        "batch_size_M": int(batch_size_M),
        "optimization_steps": int(optimization_steps),
        "num_log_iter": int(num_log_iter),
        "geology_files": [str(g) for g in geology_files],
        "geology_metadata": [
            {
                "geology_file": str(geology_files[k]),
                "geology_name": geology_names[k],
                "geology_config_id": geology_metadata[k].get("geology_config_id"),
                "geology_rep_num": geology_metadata[k].get("rep_num"),
                "geology_scenario_name": geology_metadata[k].get("scenario_name"),
                "geology_sample_num": geology_metadata[k].get("sample_num"),
                "geology_data_format_version": geology_metadata[k].get(
                    "data_format_version"
                ),
                "geology_model_version": geology_metadata[k].get("model_version"),
            }
            for k in range(len(geology_files))
        ],
        "revenue_model": {
            "checkpoint": str(revenue_checkpoint_path),
            "scaler": str(revenue_scaler_path),
            "target": revenue_target,
        },
        "energy_model": {
            "checkpoint": str(energy_checkpoint_path),
            "scaler": str(energy_scaler_path),
            "target": energy_target,
        },
        "summary_plots": [
            str(plot_path1),
            str(plot_path2),
            str(plot_path3),
            str(plot_path4),
        ],
        "results_csv": str(csv_path),
        "snapshot_count": len(snapshot_records),
        "snapshots": snapshot_records,
    }
    with open(snapshot_manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
    tqdm.write(f"Saved snapshot manifest to {snapshot_manifest_path}")

    for h in open_h5_handles:
        h.close()

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    tqdm.write(f"\nTotal Script Runtime: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
