#!/usr/bin/env python3
"""Ensemble active learning with batched Adam optimization and LHS initialization."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import qmc
from torch_geometric.data import Batch
from tqdm.auto import tqdm

# Ensure repo-root modules are importable when running as `python inference/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compile_minimal_geothermal_h5 import (
    build_wells_table,
    extract_vertical_profiles,
    extract_well_data,
)
from geothermal.active_learning_utils import (
    read_geology_metadata,
    to_julia_wells_text,
)
from geothermal.data import HeteroGraphScaler, build_single_hetero_data
from geothermal.model import HeteroGNNRegressor
from preprocess_h5 import PROPERTIES, PERM_PROPS, find_z_cutoff, get_valid_mask

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
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


def _set_plot_style() -> None:
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


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(MANIM_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(MANIM_WHITE)
    ax.tick_params(colors=MANIM_WHITE, labelsize=TICK_SIZE)
    ax.xaxis.label.set_color(MANIM_WHITE)
    ax.yaxis.label.set_color(MANIM_WHITE)
    ax.title.set_color(MANIM_WHITE)


def _inverse_targets_1d(scaler, values_scaled_1d: np.ndarray) -> np.ndarray:
    flat = values_scaled_1d.reshape(-1, 1)
    return scaler.inverse_targets(flat).reshape(-1)


def _scaler_to_torch(
    scaler: HeteroGraphScaler, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(scaler.target_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.target_scaler.scale_, dtype=torch.float32, device=device)
    return mean, scale


def _inverse_target_torch(
    y_scaled: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return y_scaled * scale + mean


def _resolve_device(gpu_index: int) -> torch.device:
    if gpu_index >= 0 and torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {gpu_index} requested but only {torch.cuda.device_count()} "
                "GPU(s) are available."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def _load_ensemble_members(
    ensemble_dir: Path,
    checkpoint_glob: str,
    scaler_filename: str,
) -> list[dict[str, Path]]:
    ckpt_paths = sorted(ensemble_dir.glob(checkpoint_glob))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found under {ensemble_dir} with '{checkpoint_glob}'."
        )
    members: list[dict[str, Path]] = []
    for ckpt_path in ckpt_paths:
        scaler_path = ckpt_path.parent / scaler_filename
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        members.append({"checkpoint": ckpt_path, "scaler": scaler_path})
    return members


def _build_snapshot_payload(
    run_id: int,
    iteration: int,
    is_final_iteration: bool,
    coords_xyz: np.ndarray,
    is_injector_list: list[bool],
    geology_file: str,
    geology_name: str,
    geology_metadata: dict[str, Any],
    ensemble_member_revenues: np.ndarray,
    ensemble_labels: list[str],
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

    ensemble_predictions = []
    for idx, (label, rev) in enumerate(zip(ensemble_labels, ensemble_member_revenues)):
        ensemble_predictions.append(
            {
                "member_index": idx,
                "member_label": label,
                "discounted_total_revenue": float(rev),
            }
        )

    payload = {
        "snapshot_id": f"run{run_id:04d}_iter{iteration:04d}",
        "run_id": int(run_id),
        "iteration": int(iteration),
        "is_final_iteration": bool(is_final_iteration),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "geology_file": str(geology_file),
        "geology_name": geology_name,
        "geology_config_id": geology_metadata.get("geology_config_id"),
        "wells": wells_json,
        "ensemble_predictions": ensemble_predictions,
        "summary": {
            "mean_discounted_total_revenue": float(np.mean(ensemble_member_revenues)),
            "std_discounted_total_revenue": float(np.std(ensemble_member_revenues, ddof=0)),
        },
    }
    return payload


def main() -> None:
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Ensemble Active Learning Optimization with LHS Initialization"
    )
    parser.add_argument("--config", type=Path, required=True, help="JSON configuration")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (-1 for CPU)")
    parser.add_argument(
        "--optimization-objective",
        choices=["primary", "ensemble_mean", "ucb"],
        default=None,
    )
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config_dir = config_path.parent

    def _resolve_config_path(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (config_dir / p)

    with open(config_path, "r") as f:
        config = json.load(f)

    geology_file = str(_resolve_config_path(config["geology_h5_file"]))
    norm_config_path = _resolve_config_path(config.get("norm_config", "norm_config.json"))
    with open(norm_config_path, "r") as f:
        norm_config = json.load(f)

    ensemble_dir = _resolve_config_path(config["ensemble_dir"])
    checkpoint_glob = config.get("checkpoint_glob", "run_*/checkpoints/best-*.ckpt")
    scaler_filename = config.get("scaler_filename", "scaler.pkl")
    primary_member = int(config.get("primary_member", 0))

    num_samples_N = int(config.get("num_samples_N", 50))
    batch_size_M = int(config.get("batch_size_M", 10))
    edge_buffer = float(config.get("edge_buffer", 5.0))
    log_every_n_steps = int(config.get("log_every_n_steps", 25))
    optimization_steps = int(config.get("optimization_steps", 100))
    learning_rate = float(config.get("learning_rate", 0.5))
    ucb_beta = float(config.get("ucb_beta", 1.0))
    opt_objective = (
        args.optimization_objective
        if args.optimization_objective
        else config.get("optimization_objective", "primary")
    )
    revenue_target = config.get("revenue_target", "graph_discounted_net_revenue")

    device = _resolve_device(args.gpu)
    tqdm.write(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else _resolve_config_path(config.get("output_root", "inference_outputs/ensemble_active_learning"))
    )
    tag = opt_objective
    if opt_objective == "ucb":
        tag = f"ucb_beta{ucb_beta:g}"
    run_output_dir = output_root / f"ensemble_{timestamp}_{tag}"
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

    tqdm.write("Loading ensemble members...")
    members = _load_ensemble_members(ensemble_dir, checkpoint_glob, scaler_filename)
    if primary_member < 0 or primary_member >= len(members):
        raise ValueError(f"primary_member {primary_member} out of range.")

    member_labels = [f"member_{i:02d}" for i in range(len(members))]

    member_payloads = []
    for idx, member in enumerate(members):
        with member["scaler"].open("rb") as f:
            member_scaler: HeteroGraphScaler = pickle.load(f)
        member_model = HeteroGNNRegressor.load_from_checkpoint(
            str(member["checkpoint"]), map_location=device
        ).to(device)
        member_model.eval()
        target_mean, target_scale = _scaler_to_torch(member_scaler, device)
        member_payloads.append(
            {
                "scaler": member_scaler,
                "model": member_model,
                "target_mean": target_mean,
                "target_scale": target_scale,
            }
        )

    tqdm.write(f"Loaded {len(members)} models.")

    # Load base geology
    with h5py.File(geology_file, "r") as src:
        geo_name = Path(geology_file).stem
        geo_meta = read_geology_metadata(src, geology_name=geo_name)
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
                (data - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(data)
            )
            normalized = np.clip(normalized, 0.0, 1.0)
            normalized[~valid_mask_cropped] = 0.0
            physics_dict[prop] = torch.tensor(normalized, dtype=torch.float32)

        perm_x_grid = physics_dict["PermX"].clone().cpu().numpy()
        physics_dict["valid_mask"] = torch.tensor(valid_mask_cropped, dtype=torch.float32)
        full_shape = (z_cutoff, valid_mask.shape[1], valid_mask.shape[2])
        z_max, nx, ny = full_shape

    z_slice = perm_x_grid.shape[0] // 2
    background = perm_x_grid[z_slice, :, :].T

    num_wells = len(config["wells"])
    is_injector_list = []
    for w in range(num_wells):
        w_type = config["wells"][w].get("type", "injector").lower()
        is_injector_list.append(w_type == "injector")

    # Latin Hypercube Sampling
    tqdm.write(f"Generating {num_samples_N} LHS samples...")
    x_min_bound, x_max_bound = edge_buffer, nx - 1 - edge_buffer
    y_min_bound, y_max_bound = edge_buffer, ny - 1 - edge_buffer
    sampler = qmc.LatinHypercube(d=2 * num_wells, seed=42)
    lhs_samples = sampler.random(n=num_samples_N)

    all_cfgs = []
    for n in range(num_samples_N):
        c = []
        for w in range(num_wells):
            rx = x_min_bound + lhs_samples[n, 2 * w] * (x_max_bound - x_min_bound)
            ry = y_min_bound + lhs_samples[n, 2 * w + 1] * (y_max_bound - y_min_bound)
            depth = min(config["wells"][w].get("depth", z_max), z_max)
            w_type = config["wells"][w].get("type", "injector").lower()
            c.append({"x": rx, "y": ry, "depth": depth, "type": w_type})
        all_cfgs.append(c)

    # Rendering Helper
    def _save_snapshot_figure(
        snapshot_payload: dict[str, Any], fig_path: Path
    ) -> None:
        _set_plot_style()
        fig, ax_map = plt.subplots(figsize=(8, 7), facecolor=MANIM_BG)
        _style_ax(ax_map)
        im = ax_map.imshow(
            background,
            origin="lower",
            cmap="viridis",
            alpha=0.55,
            extent=[0, nx, 0, ny],
        )
        cbar = plt.colorbar(im, ax=ax_map)
        cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
        cbar.set_label("Normalized Log PermX", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE)
        cbar.outline.set_edgecolor(MANIM_WHITE)

        for w in snapshot_payload["wells"]:
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
        ax_map.set_ylabel("Y Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
        ax_map.set_title(
            f"{geo_name} (iter={snapshot_payload['iteration']})",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )
        fig.suptitle(
            f"Run {snapshot_payload['run_id']} Iter {snapshot_payload['iteration']}\n"
            f"Mean Rev={snapshot_payload['summary']['mean_discounted_total_revenue']:.3f} | "
            f"Std Rev={snapshot_payload['summary']['std_discounted_total_revenue']:.3f}",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )

        fig.tight_layout()
        fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
        plt.close(fig)

    snapshot_records = []

    def _log_snapshot(
        run_id: int,
        iteration: int,
        is_final_iteration: bool,
        coords_xyz: np.ndarray,
        ensemble_member_revenues: np.ndarray,
    ) -> None:
        payload = _build_snapshot_payload(
            run_id=run_id,
            iteration=iteration,
            is_final_iteration=is_final_iteration,
            coords_xyz=coords_xyz,
            is_injector_list=is_injector_list,
            geology_file=geology_file,
            geology_name=geo_name,
            geology_metadata=geo_meta,
            ensemble_member_revenues=ensemble_member_revenues,
            ensemble_labels=member_labels,
        )
        payload["predictions_by_geology"] = [
            {
                "geology_index": 0,
                "geology_name": geo_name,
                "geology_file": geology_file,
                "geology_config_id": geo_meta.get("geology_config_id"),
                "geology_scenario_name": geo_meta.get("scenario_name"),
                "geology_sample_num": geo_meta.get("sample_num"),
                "discounted_total_revenue": payload["summary"]["mean_discounted_total_revenue"],
                "total_energy_production": 0.0,
            }
        ]
        
        snapshot_id = payload["snapshot_id"]
        json_path = snapshots_json_dir / f"{snapshot_id}.json"
        fig_path = snapshots_fig_dir / f"{snapshot_id}.png"
        jl_path = snapshot_well_configs_dir / f"{snapshot_id}.jl"

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        _save_snapshot_figure(payload, fig_path)

        jl_text = to_julia_wells_text(
            coords_xyz=coords_xyz,
            is_injector_list=is_injector_list,
            score=float(payload["summary"]["mean_discounted_total_revenue"]),
            score_label="Mean Discounted Revenue",
            geology_file=geology_file,
            geology_name=geo_name,
            geology_config_id=geo_meta.get("geology_config_id"),
            geology_scenario_name=geo_meta.get("scenario_name"),
            geology_sample_num=geo_meta.get("sample_num"),
            ensemble_mean_revenue=float(payload["summary"]["mean_discounted_total_revenue"]),
            ensemble_std_revenue=float(payload["summary"]["std_discounted_total_revenue"]),
            ensemble_member_revenues=[float(r) for r in ensemble_member_revenues],
            ensemble_labels=member_labels,
        )
        with open(jl_path, "w") as f:
            f.write(jl_text)

        snapshot_records.append(
            {
                "snapshot_id": snapshot_id,
                "run_id": int(run_id),
                "iteration": int(iteration),
                "json_path": str(json_path),
                "figure_path": str(fig_path),
                "well_config_path": str(jl_path),
                "well_config_paths_by_geology": [
                    {
                        "geology_index": 0,
                        "geology_name": geo_name,
                        "geology_file": geology_file,
                        "geology_config_id": geo_meta.get("geology_config_id"),
                        "geology_scenario_name": geo_meta.get("scenario_name"),
                        "geology_sample_num": geo_meta.get("sample_num"),
                        "well_config_path": str(jl_path),
                    }
                ],
                "mean_discounted_total_revenue": payload["summary"]["mean_discounted_total_revenue"],
                "std_discounted_total_revenue": payload["summary"]["std_discounted_total_revenue"],
                "ensemble_predictions": payload["ensemble_predictions"],
            }
        )

    def _should_log_step(iteration: int) -> bool:
        if log_every_n_steps <= 0:
            return iteration == optimization_steps
        if iteration == 0:
            return True
        return (iteration % log_every_n_steps) == 0 or iteration == optimization_steps

    all_trajectories = []
    all_final_revenues = []

    # Batch Processing
    tqdm.write(f"\n--- Starting Batched Adam Pipeline ({opt_objective}) ---")
    for batch_index, chunk_start in enumerate(
        tqdm(range(0, num_samples_N, batch_size_M), desc="Batches", unit="batch"), start=1
    ):
        batch_start_time = time.time()
        M_actual = min(batch_size_M, num_samples_N - chunk_start)
        chunk_cfgs = all_cfgs[chunk_start : chunk_start + M_actual]

        # Build graphs for the batch. Since the geometry changes with pos_xyz in Adam,
        # we build the static parts here.
        chunk_member_batches = [[] for _ in members]

        with h5py.File(geology_file, "r") as src:
            temp0_full = src["Input/Temperature0"][:]
            for m_idx, w_cfg in enumerate(chunk_cfgs):
                is_well = np.zeros((z_cutoff, nx, ny), dtype=np.int32)
                inj_rate = np.zeros((z_cutoff, nx, ny), dtype=np.float32)

                for w in w_cfg:
                    ix, iy = int(round(w["x"])), int(round(w["y"]))
                    ix, iy = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
                    for z in range(int(w["depth"])):
                        if temp0_full[z, ix, iy] <= -900:
                            is_well[z, ix, iy] = -999
                            inj_rate[z, ix, iy] = -999
                        else:
                            is_well[z, ix, iy] = 1
                            inj_rate[z, ix, iy] = 8000.0 if w["type"] == "injector" else -8000.0

                (
                    x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z,
                    porosity, temp0, press0, depth_centroid,
                ) = extract_well_data(is_well, inj_rate, src)
                wells = build_wells_table(
                    x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z, porosity, temp0, press0
                )
                vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)

                raw_graph = build_single_hetero_data(
                    wells=wells,
                    physics_dict=physics_dict,
                    full_shape=full_shape,
                    target=revenue_target,
                    target_val=0.0,
                    vertical_profile=vertical_profiles,
                    case_id=f"run{chunk_start+m_idx}",
                )

                for e_idx, payload in enumerate(member_payloads):
                    chunk_member_batches[e_idx].append(
                        payload["scaler"].transform_graph(raw_graph)
                    )

        batch_pyg_data = []
        for e_idx in range(len(members)):
            batch_pyg_data.append(Batch.from_data_list(chunk_member_batches[e_idx]).to(device))

        continuous_starts = []
        for w_cfg in chunk_cfgs:
            c = []
            for w in w_cfg:
                c.append([w["x"], w["y"], float(w["depth"])])
            continuous_starts.append(c)

        base_coords = torch.tensor(continuous_starts, dtype=torch.float32, device=device)
        base_coords.requires_grad = True

        optimizer = optim.Adam([base_coords], lr=learning_rate)
        chunk_coords_hist = [base_coords.clone().detach().cpu().numpy()]

        def _predict_all(coords_tensor: torch.Tensor) -> np.ndarray:
            expanded = coords_tensor.view(-1, 3)
            all_preds = []
            for e_idx, payload in enumerate(member_payloads):
                batch_pyg_data[e_idx]["well"].pos_xyz = expanded
                pred_scaled = payload["model"](batch_pyg_data[e_idx]).view(M_actual)
                pred_unscaled = _inverse_target_torch(
                    pred_scaled, payload["target_mean"], payload["target_scale"]
                )
                all_preds.append(pred_unscaled.detach().cpu().numpy())
            return np.stack(all_preds, axis=1)  # (M, E)

        if _should_log_step(0):
            with torch.no_grad():
                preds = _predict_all(base_coords)
            for m in range(M_actual):
                run_id = chunk_start + m
                _log_snapshot(run_id, 0, False, base_coords[m].detach().cpu().numpy(), preds[m])

        steps_iter = tqdm(
            range(optimization_steps),
            desc=f"Batch {batch_index} optimize",
            unit="step",
            leave=False,
        )
        for step in steps_iter:
            optimizer.zero_grad()
            expanded_coords = base_coords.view(-1, 3)
            
            member_preds_unscaled = []
            if opt_objective == "primary":
                # Only evaluate the primary member
                batch_pyg_data[primary_member]["well"].pos_xyz = expanded_coords
                pred_scaled = member_payloads[primary_member]["model"](batch_pyg_data[primary_member]).view(M_actual)
                loss = -pred_scaled.sum()
            else:
                for e_idx, payload in enumerate(member_payloads):
                    batch_pyg_data[e_idx]["well"].pos_xyz = expanded_coords
                    pred_scaled = payload["model"](batch_pyg_data[e_idx]).view(M_actual)
                    pred_unscaled = _inverse_target_torch(
                        pred_scaled, payload["target_mean"], payload["target_scale"]
                    )
                    member_preds_unscaled.append(pred_unscaled)

                preds_tensor = torch.stack(member_preds_unscaled, dim=1)  # (M, E)
                mean_pred = preds_tensor.mean(dim=1)
                if opt_objective == "ensemble_mean":
                    loss = -mean_pred.sum()
                else:
                    std_pred = preds_tensor.std(dim=1, unbiased=False)
                    obj = mean_pred + ucb_beta * std_pred
                    loss = -obj.sum()

            loss.backward()

            gradients = base_coords.grad  # (M, W, 3)
            with torch.no_grad():
                for d, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                    mask_lower = (base_coords[:, :, d] <= 1e-4) & (gradients[:, :, d] > 0)
                    mask_upper = (base_coords[:, :, d] >= max_val - 1e-4) & (gradients[:, :, d] < 0)
                    mask_out = mask_lower | mask_upper
                    if mask_out.any():
                        gradients[..., d][mask_out] = 0.0
                        if base_coords in optimizer.state and "exp_avg" in optimizer.state[base_coords]:
                            optimizer.state[base_coords]["exp_avg"][..., d][mask_out] = 0.0

            optimizer.step()

            with torch.no_grad():
                base_coords[:, :, 0].clamp_(0, nx - 1)
                base_coords[:, :, 1].clamp_(0, ny - 1)
                base_coords[:, :, 2].clamp_(0, z_max - 1)

            chunk_coords_hist.append(base_coords.clone().detach().cpu().numpy())

            iteration = step + 1
            if _should_log_step(iteration):
                with torch.no_grad():
                    preds = _predict_all(base_coords)
                is_final = iteration == optimization_steps
                for m in range(M_actual):
                    run_id = chunk_start + m
                    _log_snapshot(run_id, iteration, is_final, base_coords[m].detach().cpu().numpy(), preds[m])

        chunk_coords_np = np.stack(chunk_coords_hist, axis=0)  # (steps+1, M, W, 3)
        chunk_coords_np = np.transpose(chunk_coords_np, (1, 0, 2, 3))  # (M, steps+1, W, 3)

        for m in range(M_actual):
            all_trajectories.append(chunk_coords_np[m])
            # Save final predictions
            all_final_revenues.append(preds[m].mean())

    tqdm.write("\n--- Optimization Complete ---")

    all_final_revenues = np.array(all_final_revenues)
    best_idx = np.argmax(all_final_revenues)
    tqdm.write(f"Top Run {best_idx} Mean Discounted Revenue: {all_final_revenues[best_idx]:.3f}")

    # ========================================================
    # SUMMARY PLOTS
    # ========================================================
    tqdm.write("\n--- Generating Summary Plots ---")
    _set_plot_style()
    
    # 1. Trajectories Plot
    fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor=MANIM_BG)
    _style_ax(ax1)
    im1 = ax1.imshow(
        background, origin="lower", cmap="viridis", alpha=0.55, extent=[0, nx, 0, ny]
    )
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Normalized Log PermX", color=MANIM_WHITE, fontsize=CBAR_LABEL_SIZE)
    cbar1.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cbar1.outline.set_edgecolor(MANIM_WHITE)
    
    for n_idx in range(num_samples_N):
        history = all_trajectories[n_idx]
        for w in range(num_wells):
            is_inj = is_injector_list[w]
            color = MANIM_BLUE if is_inj else MANIM_ORANGE
            marker = "^" if is_inj else "v"
            ax1.scatter(
                history[-1, w, 0], history[-1, w, 1], color=color, marker=marker, s=50, alpha=0.6,
                edgecolors=MANIM_WHITE, linewidths=0.5
            )
            ax1.plot(
                history[:, w, 0], history[:, w, 1], color=color, alpha=0.2, linewidth=1.0
            )

    ax1.set_title("All Final Placements & Trajectories", color=MANIM_WHITE, fontsize=TITLE_SIZE)
    ax1.set_xlabel("X Coordinate", color=MANIM_WHITE, fontsize=FONT_SIZE)
    ax1.set_ylabel("Y Coordinate", color=MANIM_WHITE, fontsize=FONT_SIZE)
    fig1.tight_layout()
    plot_path1 = summary_plots_dir / f"ensemble_trajectories_{timestamp}.png"
    fig1.savefig(plot_path1, dpi=200, facecolor=MANIM_BG)
    plt.close(fig1)

    # 2. Heatmaps
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), facecolor=MANIM_BG)
    _style_ax(axes2[0])
    _style_ax(axes2[1])

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

    hb_inj = axes2[0].hexbin(
        final_inj_x, final_inj_y, gridsize=20, cmap="Blues", extent=[0, nx, 0, ny], mincnt=1, alpha=0.95
    )
    cb_inj = plt.colorbar(hb_inj, ax=axes2[0])
    cb_inj.set_label("Count", color=MANIM_WHITE, fontsize=CBAR_LABEL_SIZE)
    cb_inj.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cb_inj.outline.set_edgecolor(MANIM_WHITE)
    axes2[0].set_title("Injector Final Placements", color=MANIM_WHITE, fontsize=TITLE_SIZE)

    hb_prd = axes2[1].hexbin(
        final_prd_x, final_prd_y, gridsize=20, cmap="Oranges", extent=[0, nx, 0, ny], mincnt=1, alpha=0.95
    )
    cb_prd = plt.colorbar(hb_prd, ax=axes2[1])
    cb_prd.set_label("Count", color=MANIM_WHITE, fontsize=CBAR_LABEL_SIZE)
    cb_prd.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cb_prd.outline.set_edgecolor(MANIM_WHITE)
    axes2[1].set_title("Producer Final Placements", color=MANIM_WHITE, fontsize=TITLE_SIZE)

    for ax in axes2:
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_xlabel("X Coordinate", color=MANIM_WHITE, fontsize=FONT_SIZE)
        ax.set_ylabel("Y Coordinate", color=MANIM_WHITE, fontsize=FONT_SIZE)

    fig2.tight_layout()
    plot_path2 = summary_plots_dir / f"ensemble_heatmaps_{timestamp}.png"
    fig2.savefig(plot_path2, dpi=200, facecolor=MANIM_BG)
    plt.close(fig2)

    # Results CSV
    csv_path = run_output_dir / f"ensemble_cmaes_{timestamp}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["run_id", "well_id", "well_type", "depth_z", "final_x", "final_y", "mean_discounted_total_revenue", "std_discounted_total_revenue"]
        header.extend([f"{label}_revenue" for label in member_labels])
        writer.writerow(header)
        
        final_snapshots = {s["run_id"]: s for s in snapshot_records if s["iteration"] == optimization_steps}
        
        for n_idx in range(num_samples_N):
            history = all_trajectories[n_idx]
            final_coords = history[-1]
            s_rec = final_snapshots.get(n_idx, {})
            mean_rev = s_rec.get("mean_discounted_total_revenue", all_final_revenues[n_idx])
            std_rev = s_rec.get("std_discounted_total_revenue", 0.0)
            member_revs = [p["discounted_total_revenue"] for p in s_rec.get("ensemble_predictions", [])]
            
            for w in range(num_wells):
                x, y, z = final_coords[w]
                w_type = "injector" if is_injector_list[w] else "producer"
                row = [n_idx, w, w_type, z, x, y, mean_rev, std_rev]
                row.extend(member_revs)
                writer.writerow(row)

    tqdm.write(f"Results saved to {csv_path}")

    # Snapshot Manifest
    snapshot_manifest_path = run_output_dir / f"snapshot_manifest_{timestamp}.json"
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "objective": opt_objective,
        "num_samples_N": int(num_samples_N),
        "batch_size_M": int(batch_size_M),
        "optimization_steps": int(optimization_steps),
        "log_every_n_steps": int(log_every_n_steps),
        "geology_file": str(geology_file),
        "geology_metadata": {
            "geology_file": str(geology_file),
            "geology_name": geo_name,
            "geology_config_id": geo_meta.get("geology_config_id"),
            "geology_rep_num": geo_meta.get("rep_num"),
            "geology_scenario_name": geo_meta.get("scenario_name"),
            "geology_sample_num": geo_meta.get("sample_num"),
            "geology_data_format_version": geo_meta.get("data_format_version"),
            "geology_model_version": geo_meta.get("model_version"),
        },
        "ensemble": {
            "ensemble_dir": str(ensemble_dir),
            "checkpoint_glob": checkpoint_glob,
            "scaler_filename": scaler_filename,
            "member_count": len(members),
            "primary_member": primary_member,
        },
        "summary_plots": [
            str(plot_path1),
            str(plot_path2),
        ],
        "results_csv": str(csv_path),
        "snapshot_count": len(snapshot_records),
        "snapshots": snapshot_records,
    }
    with open(snapshot_manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
    tqdm.write(f"Saved snapshot manifest to {snapshot_manifest_path}")

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    tqdm.write(f"\nTotal Script Runtime: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
