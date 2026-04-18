#!/usr/bin/env python3
"""
Differentiable Inference Wrapper for Well Location Optimization (LHS Batch Geologies)

This script loads a trained HeteroGNNRegressor model and multiple
geothermal case graphs. It generates N initial well coordinate sets via Latin Hypercube
Sampling natively across the continuous space, bounded by a specified edge buffer.

The optimization process evaluates combinations iteratively across compute chunks (size M),
running parallel optimization of the contiguous well coordinates simultaneously
across ALL candidate geologies to find a robust mean energy yield under variance bounds.
"""

from __future__ import annotations

import argparse
import json
import csv
import time
import logging
from pathlib import Path
import os
import pickle
from datetime import datetime

import h5py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
        "--optimization-steps",
        type=int,
        default=50,
        help="Number of gradient ascent steps",
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
        "--export-jl",
        type=str,
        default="",
        help="Path to export the optimized well placement in Julia configuration format.",
    )
    args = parser.parse_args()

    # 1. Load Configurations & Device Setting
    with open(args.config, "r") as f:
        config = json.load(f)

    checkpoint_path = Path(config.get("checkpoint"))
    if not checkpoint_path:
        raise ValueError("Missing 'checkpoint' path in JSON configuration")
    scaler_path = Path(config.get("scaler_path"))
    if not scaler_path:
        raise ValueError("Missing 'scaler_path' in JSON configuration")

    geology_files = config.get("geology_h5_files", [config.get("geology_h5_file")])
    if not geology_files or geology_files[0] is None:
        raise ValueError(
            "Missing 'geology_h5_files' or 'geology_h5_file' in configuration"
        )

    norm_config_path = Path(config.get("norm_config", "norm_config.json"))
    if not norm_config_path.exists():
        raise FileNotFoundError(f"Norm config not found: {norm_config_path}")

    with open(norm_config_path, "r") as f:
        norm_config = json.load(f)

    num_samples_N = config.get("num_samples_N", 30)
    batch_size_M = config.get("batch_size_M", 10)
    edge_buffer = config.get("edge_buffer", 5)

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
    print(f"Using device: {device}")

    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded training scaler from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    if checkpoint_path.exists():
        model = HeteroGNNRegressor.load_from_checkpoint(
            str(checkpoint_path), map_location=device
        )
        print(f"Loaded trained model checkpoint from {checkpoint_path}.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # 2. Preload Base Geologies to Memory Space
    open_h5_handles = [h5py.File(g, "r") for g in geology_files]
    h5_physics_contexts = []

    first_perm_x_grid = None
    z_max, nx, ny = 0, 0, 0

    print("Pre-loading base geology tensors...")
    for src in open_h5_handles:
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

    # Plot base arrays
    perm_x_grids = [
        p["physics_dict"]["PermX"].cpu().numpy() for p in h5_physics_contexts
    ]
    z_slice = perm_x_grids[0].shape[0] // 2
    backgrounds = [grid[z_slice, :, :].T for grid in perm_x_grids]

    # 3. Optuna CMA-ES Study Setup
    num_wells = len(config["wells"])
    print(
        f"Initializing Optuna CMA-ES for {num_samples_N} trials targeting {num_wells} wells..."
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
    all_final_energies = []  # (N,) length array tracing final mean energies
    all_trajectories = []  # List of arrays [steps+1, W, 3]
    all_energy_tracks = []  # List of arrays [steps+1, K]

    K = len(geology_files)

    print(f"\n--- Starting Batched Global-Local Optuna Pipeline ---")
    for chunk_start in range(0, num_samples_N, batch_size_M):
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
        chunk_graphs = []

        print(
            f"Loading/Building PyG models for batch {chunk_start // batch_size_M + 1} ({M_actual} configurations evaluated structurally over {K} geologies)..."
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

                raw_graph = build_single_hetero_data(
                    wells=wells,
                    physics_dict=p_ctx["physics_dict"],
                    full_shape=p_ctx["full_shape"],
                    target="graph_energy_total",
                    target_val=0.0,
                    vertical_profile=vertical_profiles,
                    case_id=f"run{chunk_start+m_idx}_geo{k_idx}",
                )
                chunk_graphs.append(scaler.transform_graph(raw_graph))

        batch = Batch.from_data_list(chunk_graphs).to(device)

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
        chunk_energy_hist = []  # list of (M, K)

        print(f"Optimizing chunk for {args.optimization_steps} iterations...")
        for step in range(args.optimization_steps):
            optimizer.zero_grad()

            expanded_coords = base_coords.unsqueeze(1).repeat(1, K, 1, 1).view(-1, 3)
            batch["well"].pos_xyz = expanded_coords

            predicted_energy = model(batch)  # (M * K, 1)
            pred_split = predicted_energy.view(M_actual, K)
            chunk_energy_hist.append(pred_split.clone().detach().cpu().numpy())

            # Gradients cleanly distribute independently!
            loss = -predicted_energy.sum()
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

        # Extract Terminal Output
        with torch.no_grad():
            expanded_coords = base_coords.unsqueeze(1).repeat(1, K, 1, 1).view(-1, 3)
            batch["well"].pos_xyz = expanded_coords
            final_pred = model(batch).view(M_actual, K).detach().cpu().numpy()
            chunk_energy_hist.append(final_pred)

        chunk_coords_np = np.stack(chunk_coords_hist, axis=0)  # (steps+1, M, W, 3)
        chunk_coords_np = np.transpose(
            chunk_coords_np, (1, 0, 2, 3)
        )  # (M, steps+1, W, 3)

        chunk_energy_np = np.stack(chunk_energy_hist, axis=0)  # (steps+1, M, K)
        chunk_energy_np = np.transpose(chunk_energy_np, (1, 0, 2))  # (M, steps+1, K)

        for m in range(M_actual):
            all_trajectories.append(chunk_coords_np[m])
            flat_energies = chunk_energy_np[m].reshape(-1, 1)
            unnorm_flat = scaler.inverse_targets(flat_energies).flatten()
            unnorm = unnorm_flat.reshape(-1, K)

            all_energy_tracks.append(unnorm)
            final_mean_energy = unnorm[-1].mean()
            all_final_energies.append(final_mean_energy)

            # Baldwinian Step: Give Optuna the Local Optimized Energy
            study.tell(chunk_trials[m], float(final_mean_energy))

        batch_elapsed = time.time() - batch_start_time
        print(
            f"Computed batch {chunk_start // batch_size_M + 1} / {int(np.ceil(num_samples_N / batch_size_M))} in {batch_elapsed:.2f}s"
        )

    print("\n--- Optimization Complete ---")

    # 5. Output Identifications
    all_final_energies = np.array(all_final_energies)
    best_idx = np.argmax(all_final_energies)
    worst_idx = np.argmin(all_final_energies)
    sorted_idx = np.argsort(all_final_energies)
    avg_idx = sorted_idx[len(all_final_energies) // 2]

    print(f"Top    Run {best_idx} Output Energy: {all_final_energies[best_idx]:.3f}")
    print(f"Mid    Run {avg_idx} Output Energy: {all_final_energies[avg_idx]:.3f}")
    print(f"Worst  Run {worst_idx} Output Energy: {all_final_energies[worst_idx]:.3f}")

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

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================
    # FIGURE 1: ALL RUNS COMBINED
    # ========================================================
    print("\n--- Generating Figure 1 (All Runs Map) ---")
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

    steps_range = range(args.optimization_steps + 1)
    for n_idx in range(num_samples_N):
        mean_track = all_energy_tracks[n_idx].mean(axis=1)
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
        label="Mean Batch Energy",
    )
    ax_energy1.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy1.set_ylabel(
        "Total Energy (Non-Norm)", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy1.set_title(
        f"Energy Traces across {num_samples_N} CMA-ES Samples",
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
    plot_path1 = f"plots/well_trajectories_cmaes_{timestamp}_all.png"
    fig1.savefig(plot_path1, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig1)

    # ========================================================
    # FIGURE 2: BEST, WORST, AVERAGE HIGHLIGHTS
    # ========================================================
    print("--- Generating Figure 2 (Selections Highlights Map) ---")
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
        mean_track = all_energy_tracks[idx].mean(axis=1)  # trace mean across geologies
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
        "Total Energy (Non-Norm)", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy2.set_title(
        "Energy Traces (Highlights)", fontsize=TITLE_SIZE, color=MANIM_WHITE
    )
    ax_energy2.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)
    ax_energy2.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    plt.tight_layout()
    plot_path2 = f"plots/well_trajectories_cmaes_{timestamp}_highlights.png"
    fig2.savefig(plot_path2, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig2)

    # ========================================================
    # FIGURE 3: HEATMAP OVERALL DISTRIBUTION
    # ========================================================
    print("--- Generating Figure 3 (Optimization Layout Heatmaps) ---")
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
    plot_path3 = f"plots/well_trajectories_cmaes_{timestamp}_heatmap.png"
    fig3.savefig(plot_path3, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig3)

    print(f"Saved CMA-ES All Runs plot to {plot_path1}")
    print(f"Saved CMA-ES Highlights plot to {plot_path2}")
    print(f"Saved CMA-ES Heatmap distribution to {plot_path3}")

    # ========================================================
    # FIGURE 4: OPTUNA OPTIMIZATION HISTORY
    # ========================================================
    print("--- Generating Figure 4 (Optuna Convergence History) ---")
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
            
    plot_path4 = f"plots/well_trajectories_cmaes_{timestamp}_history.png"
    fig4.tight_layout()
    fig4.savefig(plot_path4, dpi=200, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig4)
    print(f"Saved CMA-ES Optuna History to {plot_path4}")

    if args.export_jl:
        final_coords = all_trajectories[best_idx][-1]
        with open(args.export_jl, "w") as f:
            f.write("# Auto-generated by Differentiable Inference Proxy\n")
            f.write(f"# Top CMA-ES Score: {all_final_energies[best_idx]:.3f}\n")
            f.write("wells = [\n")
            for w in range(num_wells):
                x, y, z = final_coords[w]
                j_idx = int(round(float(x))) + 1
                i_idx = int(round(float(y))) + 1
                k_idx = int(round(float(z))) + 1

                is_inj = is_injector_list[w]
                well_type = '"INJECTOR"' if is_inj else '"PRODUCER"'
                rate = 8000.0 if is_inj else -8000.0

                f.write(f"    ({i_idx}, {j_idx}, {k_idx}, {well_type}, {rate}),\n")
            f.write("]\n")
        print(f"Exported optimized TOP well configuration to {args.export_jl}")

    csv_path = f"plots/well_trajectories_cmaes_{timestamp}_results.csv"
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
                "mean_total_energy",
            ]
        )
        for n_idx in range(num_samples_N):
            history = all_trajectories[n_idx]
            mean_energy = all_final_energies[n_idx]
            final_coords = history[-1]
            for w in range(num_wells):
                x, y, z = final_coords[w]
                w_type = "injector" if is_injector_list[w] else "producer"
                writer.writerow([n_idx, w, w_type, z, x, y, mean_energy])
    print(f"Exported complete final configuration tables to {csv_path}")

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"\nTotal Script Runtime: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()
