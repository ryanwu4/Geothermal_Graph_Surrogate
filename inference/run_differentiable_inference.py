#!/usr/bin/env python3
"""
Differentiable Inference Wrapper for Well Location Optimization

This script loads a trained (or untrained) HeteroGNNRegressor model and a single
geothermal case graph. It sets the continuous well (X, Y, Z) coordinates to strictly
require gradients, passes the complete structural representations through the
Continuous 3D Cropper and 3D CNN Edge Extractor, and computes `loss.backward()`.
This explicitly validates that the entire physics interpolator + CNN -> GNN pathway
is natively differentiable with respect to spatial node coordinates.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import os
import pickle
from datetime import datetime

# Ensure repo-root modules are importable when running as `python inference/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from compile_minimal_geothermal_h5 import (
    extract_well_data,
    build_wells_table,
    extract_vertical_profiles,
)
from preprocess_h5 import get_valid_mask, find_z_cutoff, PROPERTIES, PERM_PROPS
from geothermal.data import HeteroGraphScaler, build_single_hetero_data
from geothermal.model import HeteroGNNRegressor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Well Placement Differentiable Optimization"
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
        default=10,
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

    # 1. Load Configurations
    with open(args.config, "r") as f:
        config = json.load(f)

    checkpoint_path = Path(config.get("checkpoint"))
    if not checkpoint_path:
        raise ValueError("Missing 'checkpoint' path in JSON configuration")
    scaler_path = Path(config.get("scaler_path"))
    if not scaler_path:
        raise ValueError("Missing 'scaler_path' in JSON configuration")
    # Target is informational here — the loaded checkpoint determines what the
    # model actually predicts. We thread this through to build_single_hetero_data
    # so case_id / prediction_level metadata is consistent with the checkpoint.
    target = config.get("target", "graph_discounted_net_revenue")

    geology_files = config.get("geology_h5_files", [config.get("geology_h5_file")])
    if not geology_files or geology_files[0] is None:
        raise ValueError(
            "Missing 'geology_h5_files' or 'geology_h5_file' in configuration"
        )

    geology_paths = [Path(g) for g in geology_files]
    for gp in geology_paths:
        if not gp.exists():
            raise FileNotFoundError(f"Geology H5 file not found: {gp}")

    norm_config_path = Path(config.get("norm_config", "norm_config.json"))
    if not norm_config_path.exists():
        raise FileNotFoundError(f"Norm config not found: {norm_config_path}")

    with open(norm_config_path, "r") as f:
        norm_config = json.load(f)

    # 2. Setup Scaler
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded training scaler from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    # 3. Process all Geology H5 files & map wells natively
    print(f"Loading raw geology data from {len(geology_paths)} geology file(s)...")
    target_graphs = []
    backgrounds = []
    geology_names = []
    grid_shapes = []

    for geo_idx, geology_path in enumerate(geology_paths):
        print(f"  - Geology {geo_idx + 1}/{len(geology_paths)}: {geology_path}")
        with h5py.File(geology_path, "r") as src:
            valid_mask = get_valid_mask(src)
            z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
            valid_mask_cropped = valid_mask[:z_cutoff]

            print(f"    Depth horizon cutoff at Z={z_cutoff}")

            # Prepare normalized tensors
            physics_dict = {}
            for prop in PROPERTIES:
                data = src[f"Input/{prop}"][:z_cutoff].astype(np.float32)
                if prop in PERM_PROPS:
                    data = np.log10(np.maximum(data, 1e-15))

                p_min = norm_config[prop]["min"]
                p_max = norm_config[prop]["max"]

                if p_max > p_min:
                    normalized = (data - p_min) / (p_max - p_min)
                else:
                    normalized = np.zeros_like(data)

                normalized = np.clip(normalized, 0.0, 1.0)
                normalized[~valid_mask_cropped] = 0.0
                physics_dict[prop] = torch.tensor(normalized, dtype=torch.float32)

            physics_dict["valid_mask"] = torch.tensor(
                valid_mask_cropped, dtype=torch.float32
            )
            full_shape = (z_cutoff, valid_mask.shape[1], valid_mask.shape[2])
            grid_shapes.append(full_shape)

            temp0_full = src["Input/Temperature0"][:]
            nx = full_shape[1]
            ny = full_shape[2]

            is_well = np.zeros((z_cutoff, nx, ny), dtype=np.int32)
            inj_rate = np.zeros((z_cutoff, nx, ny), dtype=np.float32)

            for w in config["wells"]:
                x_raw, y_raw = w["x"], w["y"]
                x = int(round(float(x_raw)))
                y = int(round(float(y_raw)))
                w_type = w.get("type", "injector").lower()
                if x < 0 or x >= nx or y < 0 or y >= ny:
                    print(
                        f"Warning: Well at ({x_raw}, {y_raw}) is out of bounds "
                        f"for {geology_path.name} (Grid is {nx}x{ny})"
                    )
                    continue

                well_depth = min(int(w.get("depth", z_cutoff)), z_cutoff)

                for z in range(well_depth):
                    if temp0_full[z, x, y] <= -900:
                        is_well[z, x, y] = -999
                        inj_rate[z, x, y] = -999
                    else:
                        is_well[z, x, y] = 1
                        if w_type == "injector":
                            inj_rate[z, x, y] = 8000.0
                        else:
                            inj_rate[z, x, y] = -8000.0

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
            vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)

        raw_graph = build_single_hetero_data(
            wells=wells,
            physics_dict=physics_dict,
            full_shape=full_shape,
            target=target,
            target_val=0.0,
            vertical_profile=vertical_profiles,
            case_id=f"inference_custom_geo_{geo_idx}",
        )

        target_graph = scaler.transform_graph(raw_graph)
        target_graphs.append(target_graph)
        geology_names.append(geology_path.stem)

        perm_x_grid = target_graph.physics_context.d["PermX"].cpu().numpy()
        z_slice = perm_x_grid.shape[0] // 2
        backgrounds.append(perm_x_grid[z_slice, :, :].T)

    # Collate into a batch with one graph per geology
    batch = Batch.from_data_list(target_graphs)
    num_geologies = len(target_graphs)

    # 2. Setup Model
    input_dim = batch["well"].x.shape[1]
    global_dim = batch.global_attr.shape[1]

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

    if checkpoint_path.exists():
        model = HeteroGNNRegressor.load_from_checkpoint(
            str(checkpoint_path), map_location=device
        )
        print(f"Loaded trained model checkpoint from {checkpoint_path}.")
    else:
        model = HeteroGNNRegressor(
            input_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            pooling="mean",
            residual=True,
            learning_rate=1e-3,
            weight_decay=1e-2,
            loss="huber",
            prediction_level="graph",
            output_dim=1,
            active_channels=[
                "PermX",
                "PermY",
                "PermZ",
                "Porosity",
                "Temperature0",
                "Pressure0",
                "valid_mask",
            ],
            latent_edge_dim=32,
        )
        print("Instantiated untrained model for gradient pathway validation.")

    model = model.to(device)
    batch = batch.to(device)

    model.eval()

    # 3. Setup Differentiable Parameters
    # Extract the continuous node coordinates
    coords = target_graphs[0]["well"].pos_xyz.clone().detach().to(device)

    # CRITICAL: We want to optimize the spatial coordinates
    coords.requires_grad = True

    # We use an optimizer to show we can backprop and update the positions
    # (Using Adam to maximize energy)
    optimizer = optim.Adam([coords], lr=args.learning_rate)

    print(
        f"\n--- Starting Well Placement Optimization across {num_geologies} geologies ---"
    )
    print(f"Initial coordinates:\n{coords.data}")

    # Shared optimization bounds use the intersection of all geology domains.
    z_max = min(shape[0] for shape in grid_shapes)
    nx = min(shape[1] for shape in grid_shapes)
    ny = min(shape[2] for shape in grid_shapes)

    coords_history = []
    energy_history_per_geo = []

    for step in range(args.optimization_steps):
        coords_history.append(coords.clone().detach().cpu().numpy())
        optimizer.zero_grad()

        # Broadcast one coordinate tensor across all geology graphs.
        expanded_coords = coords.unsqueeze(0).repeat(num_geologies, 1, 1).view(-1, 3)
        batch["well"].pos_xyz = expanded_coords

        # Forward Pass! -> physics cropper -> 3D CNN -> HeteroGNN
        predicted_energy = model(batch).view(num_geologies, -1).squeeze(-1)
        energy_history_per_geo.append(predicted_energy.detach().cpu().numpy())
        mean_energy = predicted_energy.mean()

        # We want to MAXIMIZE mean energy across geologies.
        loss = -mean_energy

        # Backward Pass!
        loss.backward()

        gradients = coords.grad
        print(
            f"Original Step {step} Mean Energy: {mean_energy.item():.4f} "
            f"Max Grad: {gradients.abs().max().item():.4f}"
        )
        # print(f"\nOptimization Step {step + 1}/{args.optimization_steps}")
        # print(f"Predicted Energy: {predicted_energy.item():.4f}")
        # print(f"Gradients w.r.t (X, Y, Z):\n{gradients}")

        # Project gradients along feasible directions (Feasible Direction Method)
        # Prevents Adam from accumulating unfeasible momentum pointing outside the grid.
        with torch.no_grad():
            for d, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                mask_lower = (coords[:, d] <= 1e-4) & (gradients[:, d] > 0)
                mask_upper = (coords[:, d] >= max_val - 1e-4) & (gradients[:, d] < 0)
                mask_out = mask_lower | mask_upper

                if mask_out.any():
                    # Nullify the gradient component pointing outside
                    gradients[:, d][mask_out] = 0.0
                    # Clear accumulated Adam momentum buffer
                    if (
                        coords in optimizer.state
                        and "exp_avg" in optimizer.state[coords]
                    ):
                        optimizer.state[coords]["exp_avg"][:, d][mask_out] = 0.0

        # Make a gradient step
        optimizer.step()

        # Clamp coordinates to ensure wells do not drift outside the domain bounds
        # Note: coords format is [X, Y, Z] while full_shape is [Z, X, Y]
        with torch.no_grad():
            coords[:, 0].clamp_(0, nx - 1)  # X bound
            coords[:, 1].clamp_(0, ny - 1)  # Y bound
            coords[:, 2].clamp_(0, z_max - 1)  # Z bound

    coords_history.append(coords.clone().detach().cpu().numpy())
    with torch.no_grad():
        expanded_coords = coords.unsqueeze(0).repeat(num_geologies, 1, 1).view(-1, 3)
        batch["well"].pos_xyz = expanded_coords
        final_energy = model(batch).view(num_geologies, -1).squeeze(-1)
        energy_history_per_geo.append(final_energy.detach().cpu().numpy())

    print("\n--- Optimization Complete ---")
    print(f"Updated Continuous Coordinates:\n{coords.data}")

    # 4. Decode Outputs and Plot Trajectories (2D with background)
    print("\n--- Generating 2D Trajectory, Energy Plots, and Animation ---")

    energy_matrix = np.stack(energy_history_per_geo, axis=0)  # [steps+1, K]
    energy_unnorm = scaler.inverse_targets(energy_matrix.reshape(-1, 1)).reshape(
        energy_matrix.shape
    )
    energy_mean_unnorm = energy_unnorm.mean(axis=1)
    energy_min_all = min(float(energy_unnorm.min()), float(energy_mean_unnorm.min()))
    energy_max_all = max(float(energy_unnorm.max()), float(energy_mean_unnorm.max()))
    energy_margin = (energy_max_all - energy_min_all) * 0.12 + 1e-3

    history = np.stack(coords_history, axis=0)  # [steps+1, N_wells, 3]
    num_frames, num_wells, _ = history.shape

    is_injector = target_graphs[0]["well"].is_injector.cpu().numpy()

    # -------------------------------------------------------------------------
    # Manim-matched colour palette (mirrors scene_cnn_slab_still.py)
    # -------------------------------------------------------------------------
    # Injectors: Manim BLUE  (#58C4DD)
    # Producers: Manim ORANGE (#FF9000)
    # Permeability background: viridis (same as Manim COLOR_LO→MID→HI palette)
    MANIM_BG = "#000000"  # black background
    MANIM_BLUE = "#58C4DD"  # Manim BLUE — injectors
    MANIM_ORANGE = "#FF9000"  # Manim ORANGE — producers
    MANIM_WHITE = "#FFFFFF"
    MANIM_GREY = "#888888"

    FONT_SIZE = 18
    TITLE_SIZE = 17
    TICK_SIZE = 16
    LEGEND_SIZE = 16
    CBAR_LABEL_SIZE = 16

    # Increase export resolution without changing original layout proportions.
    PNG_DPI = 220
    GIF_DPI = 140
    PLOT_WSPACE = 0.12

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            # Global dark-theme defaults so all text/spines render white-on-black
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

    # -------------------------------------------------------------------------
    # Helper: apply dark-theme styling to a single axes
    # -------------------------------------------------------------------------
    def _style_ax(ax):
        ax.set_facecolor(MANIM_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(MANIM_WHITE)
        ax.tick_params(colors=MANIM_WHITE, labelsize=TICK_SIZE)
        ax.xaxis.label.set_color(MANIM_WHITE)
        ax.yaxis.label.set_color(MANIM_WHITE)
        ax.title.set_color(MANIM_WHITE)

    # -------------------------------------------------------------------------
    # 4a. Static summary plot (PNG)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        num_geologies + 1,
        figsize=(10 * (num_geologies + 1), 8),
        gridspec_kw={"width_ratios": [1.2] * num_geologies + [1.5], "wspace": PLOT_WSPACE},
        facecolor=MANIM_BG,
    )
    ax_maps = axes[:num_geologies]
    ax_energy = axes[-1]
    for ax_map in ax_maps:
        _style_ax(ax_map)
    _style_ax(ax_energy)

    added_inj_trail = False
    added_prod_trail = False

    for w in range(num_wells):
        is_inj = is_injector[w] > 0.5
        color = MANIM_BLUE if is_inj else MANIM_ORANGE

        trail_label = None
        if is_inj and not added_inj_trail:
            trail_label = "Injector"
            added_inj_trail = True
        elif not is_inj and not added_prod_trail:
            trail_label = "Producer"
            added_prod_trail = True

        marker_end = "^" if is_inj else "v"
        for k, ax_map in enumerate(ax_maps):
            bg = backgrounds[k]
            if w == 0:
                im = ax_map.imshow(
                    bg,
                    origin="lower",
                    cmap="viridis",
                    alpha=0.55,
                    extent=[0, bg.shape[1], 0, bg.shape[0]],
                )
                cbar = plt.colorbar(im, ax=ax_map, pad=0.015)
                cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
                if k == 0:
                    cbar.set_label(
                        "Normalized Log PermX",
                        fontsize=CBAR_LABEL_SIZE,
                        color=MANIM_WHITE,
                    )
                cbar.outline.set_edgecolor(MANIM_WHITE)

            ax_map.plot(
                history[:, w, 0],
                history[:, w, 1],
                color=color,
                alpha=0.7,
                linestyle="-",
                linewidth=2,
                label=trail_label if k == 0 else None,
            )
            # Start marker (circle)
            ax_map.scatter(
                history[0, w, 0],
                history[0, w, 1],
                color=color,
                marker="o",
                s=80,
                edgecolors=MANIM_WHITE,
                linewidths=1.2,
                zorder=5,
            )
            # End marker (triangle)
            ax_map.scatter(
                history[-1, w, 0],
                history[-1, w, 1],
                color=color,
                marker=marker_end,
                s=200,
                edgecolors=MANIM_WHITE,
                linewidths=1.2,
                zorder=6,
            )

    for k, ax_map in enumerate(ax_maps):
        z_slice = grid_shapes[k][0] // 2
        ax_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
        ax_map.set_ylabel(
            "Y Coordinate" if k == 0 else "", fontsize=FONT_SIZE, color=MANIM_WHITE
        )
        ax_map.set_title(
            f"{geology_names[k]}\n{args.optimization_steps} Steps (Z={z_slice})",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )
    ax_maps[0].legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    for k in range(num_geologies):
        ax_energy.plot(
            range(len(energy_unnorm)),
            energy_unnorm[:, k],
            linewidth=1.5,
            alpha=0.45,
            color=MANIM_GREY,
            label=("Per-geology energy" if k == 0 else None),
        )
    ax_energy.plot(
        range(len(energy_mean_unnorm)),
        energy_mean_unnorm,
        marker="o",
        linewidth=2.8,
        color=MANIM_BLUE,
        markerfacecolor=MANIM_ORANGE,
        markeredgecolor=MANIM_WHITE,
        markeredgewidth=1.0,
        markersize=7,
        label="Mean energy",
    )
    ax_energy.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy.set_ylabel(
        "Total Energy Production (Non-Normalized)",
        fontsize=FONT_SIZE,
        color=MANIM_WHITE,
        labelpad=2,
    )
    ax_energy.set_title(
        "Predicted Energy Growth (Mean over Geologies)",
        fontsize=TITLE_SIZE,
        color=MANIM_WHITE,
    )
    ax_energy.set_ylim(energy_min_all - energy_margin, energy_max_all + energy_margin)
    ax_energy.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)
    ax_energy.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    fig.tight_layout(w_pad=1.2)

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/well_trajectories_2D_Energy_{timestamp}.png"
    plt.savefig(plot_path, dpi=PNG_DPI, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig)
    print(f"Saved 2D trajectory & energy plot to {plot_path}")

    # -------------------------------------------------------------------------
    # 4b. Animated GIF — wells move over time, trails grow behind them
    # -------------------------------------------------------------------------
    import io
    from PIL import Image

    GIF_FPS = 20

    print(f"Building {num_frames} animation frames (PIL fast-path, {GIF_FPS} fps)...")

    # Use per-geology extents for subplot backgrounds.
    map_extents = [(0, bg.shape[1], 0, bg.shape[0]) for bg in backgrounds]

    # Build the figure once; reuse it every frame (Agg backend renders to buffer)
    fig_anim, axes_anim = plt.subplots(
        1,
        num_geologies + 1,
        figsize=(10 * (num_geologies + 1), 8),
        gridspec_kw={"width_ratios": [1.2] * num_geologies + [1.5], "wspace": PLOT_WSPACE},
        facecolor=MANIM_BG,
    )
    ax_anim_maps = axes_anim[:num_geologies]
    ax_anim_en = axes_anim[-1]
    for ax_anim_map in ax_anim_maps:
        _style_ax(ax_anim_map)
    _style_ax(ax_anim_en)

    # Static permeability backgrounds — drawn once for each geology map.
    for k, ax_anim_map in enumerate(ax_anim_maps):
        bg = backgrounds[k]
        x0, x1, y0, y1 = map_extents[k]
        ax_anim_map.imshow(
            bg,
            origin="lower",
            cmap="viridis",
            alpha=0.55,
            extent=[x0, x1, y0, y1],
        )
        ax_anim_map.set_xlim(x0, x1)
        ax_anim_map.set_ylim(y0, y1)
        ax_anim_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
        ax_anim_map.set_ylabel(
            "Y Coordinate" if k == 0 else "",
            fontsize=FONT_SIZE,
            color=MANIM_WHITE,
        )
        z_slice = grid_shapes[k][0] // 2
        ax_anim_map.set_title(
            f"{geology_names[k]} (Z={z_slice})",
            fontsize=TITLE_SIZE,
            color=MANIM_WHITE,
        )

    ax_anim_en.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_anim_en.set_ylabel(
        "Total Energy Production\n(Non-Normalized)",
        fontsize=FONT_SIZE,
        color=MANIM_WHITE,
        labelpad=2,
    )
    ax_anim_en.set_xlim(-0.5, len(energy_mean_unnorm) - 0.5)
    ax_anim_en.set_ylim(
        energy_min_all - energy_margin, energy_max_all + energy_margin
    )
    ax_anim_en.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)

    # Pre-create all mutable artists (updated via set_data each frame)
    map_trail_lines = []
    map_dot_artists = []
    for k, ax_anim_map in enumerate(ax_anim_maps):
        trail_lines, dot_artists = [], []
        seen_labels: set[str] = set()
        for w in range(num_wells):
            is_inj = is_injector[w] > 0.5
            color = MANIM_BLUE if is_inj else MANIM_ORANGE
            label_str = "Injector" if is_inj else "Producer"
            leg_label = label_str if (k == 0 and label_str not in seen_labels) else None
            if leg_label:
                seen_labels.add(label_str)

            (trail,) = ax_anim_map.plot(
                [],
                [],
                color=color,
                alpha=0.8,
                linewidth=2,
                solid_capstyle="round",
                label=leg_label,
            )
            (dot,) = ax_anim_map.plot(
                [],
                [],
                marker="o",
                color=color,
                markersize=10,
                markeredgecolor=MANIM_WHITE,
                markeredgewidth=1.5,
                linestyle="None",
                zorder=7,
            )
            trail_lines.append(trail)
            dot_artists.append(dot)
        map_trail_lines.append(trail_lines)
        map_dot_artists.append(dot_artists)

    ax_anim_maps[0].legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
        loc="upper right",
    )

    geo_energy_lines = []
    for k in range(num_geologies):
        (geo_line,) = ax_anim_en.plot(
            [],
            [],
            color=MANIM_GREY,
            alpha=0.45,
            linewidth=1.5,
            label=("Per-geology energy" if k == 0 else None),
        )
        geo_energy_lines.append(geo_line)

    (mean_en_line,) = ax_anim_en.plot(
        [],
        [],
        color=MANIM_BLUE,
        linewidth=2.4,
        marker="o",
        markersize=5,
        markerfacecolor=MANIM_ORANGE,
        markeredgecolor=MANIM_WHITE,
        markeredgewidth=1.0,
        label="Mean energy",
    )
    en_dot = ax_anim_en.scatter(
        [],
        [],
        color=MANIM_ORANGE,
        s=120,
        edgecolors=MANIM_WHITE,
        linewidths=1.5,
        zorder=7,
    )
    title_obj = fig_anim.suptitle("", fontsize=TITLE_SIZE, color=MANIM_WHITE)
    ax_anim_en.set_title(
        "Predicted Energy Growth (Mean over Geologies)",
        fontsize=TITLE_SIZE,
        color=MANIM_WHITE,
    )
    ax_anim_en.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    fig_anim.tight_layout(w_pad=1.2)

    # ------------------------------------------------------------------
    # Fast manual render loop: draw each frame → Agg buffer → PIL Image
    # ------------------------------------------------------------------
    pil_frames: list[Image.Image] = []
    canvas = fig_anim.canvas
    canvas.draw()  # initial draw to populate the renderer

    for frame in range(num_frames):
        # Update trail + dot per well for every geology map.
        for w in range(num_wells):
            xs = history[: frame + 1, w, 0]
            ys = history[: frame + 1, w, 1]
            for k in range(num_geologies):
                map_trail_lines[k][w].set_data(xs, ys)
                map_dot_artists[k][w].set_data([xs[-1]], [ys[-1]])

        # Energy panel
        en_frame = min(frame, len(energy_mean_unnorm) - 1)
        for k in range(num_geologies):
            geo_energy_lines[k].set_data(
                range(en_frame + 1), energy_unnorm[: en_frame + 1, k]
            )
        mean_en_line.set_data(range(en_frame + 1), energy_mean_unnorm[: en_frame + 1])
        en_dot.set_offsets([[en_frame, energy_mean_unnorm[en_frame]]])

        title_obj.set_text(
            f"Well Placement Optimization — Step {frame} / {num_frames - 1} "
            f"across {num_geologies} geologies"
        )

        # Render to in-memory PNG buffer via Agg, then open with PIL
        buf = io.BytesIO()
        fig_anim.savefig(buf, format="png", dpi=GIF_DPI, facecolor=MANIM_BG)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig_anim)

    # Save GIF with PIL — hold last frame for 2 s, rest at target fps
    frame_duration_ms = int(1000 / GIF_FPS)
    durations = [frame_duration_ms] * len(pil_frames)
    durations[-1] = 2000  # pause on final frame

    gif_path = f"plots/well_optimization_{timestamp}.gif"
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"Saved optimization animation to {gif_path}")

    if args.export_jl:
        final_coords = history[-1]
        with open(args.export_jl, "w") as f:
            f.write("# Auto-generated by Differentiable Inference Proxy\n")
            f.write("wells = [\n")
            for w in range(num_wells):
                x, y, z = final_coords[w]
                # Map continuous [X, Y, Z] to 1-based [I, J, K].
                # Note: 'x' bounds natively to proxy axis 1 (nx=70), which is simulation 'j'.
                # 'y' bounds natively to proxy axis 2 (ny=76), which is simulation 'i'.
                j_idx = int(round(float(x))) + 1
                i_idx = int(round(float(y))) + 1
                k_idx = int(round(float(z))) + 1

                is_inj = is_injector[w] > 0.5
                well_type = '"INJECTOR"' if is_inj else '"PRODUCER"'
                rate = 8000.0 if is_inj else -8000.0

                f.write(f"    ({i_idx}, {j_idx}, {k_idx}, {well_type}, {rate}),\n")
            f.write("]\n")
        print(f"Exported optimized well configuration to {args.export_jl}")


if __name__ == "__main__":
    main()
