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

import os
import pickle
from datetime import datetime

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
        "--config", type=Path, required=True, help="Path to input JSON configuration file"
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
    
    geology_path = Path(config["geology_h5_file"])
    if not geology_path.exists():
        raise FileNotFoundError(f"Geology H5 file not found: {geology_path}")

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

    # 3. Process Geology H5 & Map Wells Natively
    print(f"Loading raw geology data from {geology_path}...")
    with h5py.File(geology_path, "r") as src:
        valid_mask = get_valid_mask(src)
        z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
        valid_mask_cropped = valid_mask[:z_cutoff]
        
        print(f"Depth horizon cutoff at Z={z_cutoff}")
        
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
            
        physics_dict["valid_mask"] = torch.tensor(valid_mask_cropped, dtype=torch.float32)
        full_shape = (z_cutoff, valid_mask.shape[1], valid_mask.shape[2])
        
        temp0_full = src["Input/Temperature0"][:]
        nx = full_shape[1]
        ny = full_shape[2]
        
        is_well = np.zeros((z_cutoff, nx, ny), dtype=np.int32)
        inj_rate = np.zeros((z_cutoff, nx, ny), dtype=np.float32)

        for w in config["wells"]:
            x, y = w["x"], w["y"]
            w_type = w.get("type", "injector").lower()
            if x < 0 or x >= nx or y < 0 or y >= ny:
                print(f"Warning: Well at ({x}, {y}) is out of bounds (Grid is {nx}x{ny})")
                continue
                
            well_depth = min(w.get("depth", z_cutoff), z_cutoff)

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
            x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z, 
            porosity, temp0, press0, depth_centroid
        ) = extract_well_data(is_well, inj_rate, src)
        
        wells = build_wells_table(
            x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z, 
            porosity, temp0, press0
        )
        vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)
        
    raw_graph = build_single_hetero_data(
        wells=wells,
        physics_dict=physics_dict,
        full_shape=full_shape,
        target="graph_energy_total",
        target_val=0.0,
        vertical_profile=vertical_profiles,
        case_id="inference_custom"
    )

    # Isolate a single case to optimize
    target_graph = scaler.transform_graph(raw_graph)

    # Collate into a batch of size 1
    batch = Batch.from_data_list([target_graph])

    # 2. Setup Model
    input_dim = batch["well"].x.shape[1]
    global_dim = batch.global_attr.shape[1]

    if checkpoint_path.exists():
        model = HeteroGNNRegressor.load_from_checkpoint(str(checkpoint_path))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = batch.to(device)

    model.eval()

    # 3. Setup Differentiable Parameters
    # Extract the continuous node coordinates
    coords = batch["well"].pos_xyz.clone().detach().to(device)

    # CRITICAL: We want to optimize the spatial coordinates
    coords.requires_grad = True

    # We use an optimizer to show we can backprop and update the positions
    # (Using Adam to maximize energy)
    optimizer = optim.Adam([coords], lr=args.learning_rate)

    print("\n--- Starting Well Placement Optimization ---")
    print(f"Initial coordinates:\n{coords.data}")

    # Grid limits from underlying tensor: full_shape is (Z, X, Y)
    z_max, nx, ny = target_graph.physics_context.full_shape

    coords_history = []
    energy_history = []

    for step in range(args.optimization_steps):
        coords_history.append(coords.clone().detach().cpu().numpy())
        optimizer.zero_grad()

        # Override the batch coordinates dynamically with our differentiable tensor.
        # This channels the gradients through grid_sample cleanly into `coords`.
        batch["well"].pos_xyz = coords

        # Forward Pass! -> physics cropper -> 3D CNN -> HeteroGNN
        predicted_energy = model(batch)
        energy_history.append(predicted_energy.item())

        # We want to MAXIMIZE energy, so we minimize the negative energy
        loss = -predicted_energy.sum()

        # Backward Pass!
        loss.backward()

        gradients = coords.grad
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
                    if coords in optimizer.state and 'exp_avg' in optimizer.state[coords]:
                        optimizer.state[coords]['exp_avg'][:, d][mask_out] = 0.0

        # Make a gradient step
        optimizer.step()

        # Clamp coordinates to ensure wells do not drift outside the domain bounds
        # Note: coords format is [X, Y, Z] while full_shape is [Z, X, Y]
        with torch.no_grad():
            coords[:, 0].clamp_(0, nx - 1)     # X bound
            coords[:, 1].clamp_(0, ny - 1)     # Y bound
            coords[:, 2].clamp_(0, z_max - 1)  # Z bound

    coords_history.append(coords.clone().detach().cpu().numpy())
    with torch.no_grad():
        final_energy = model(batch)
        energy_history.append(final_energy.item())

    print("\n--- Optimization Complete ---")
    print(f"Updated Continuous Coordinates:\n{coords.data}")

    # 4. Decode Outputs and Plot Trajectories (2D with background)
    print("\n--- Generating 2D Trajectory and Energy Plots ---")

    energy_array = np.array(energy_history).reshape(-1, 1)
    energy_unnorm = scaler.inverse_targets(energy_array).flatten()

    history = np.stack(coords_history, axis=0)  # [steps+1, N_wells, 3]
    num_steps, num_wells, _ = history.shape

    is_injector = batch["well"].is_injector.cpu().numpy()

    
    perm_x_grid = target_graph.physics_context.d["PermX"].cpu().numpy()
    z_slice = perm_x_grid.shape[0] // 2
    background = perm_x_grid[z_slice, :, :].T  # Transpose to (Y, X) for imshow

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    ax_map = axes[0]
    ax_energy = axes[1]

    # Plot background (extent is [xmin, xmax, ymin, ymax])
    im = ax_map.imshow(
        background,
        origin="lower",
        cmap="viridis",
        alpha=0.5,
        extent=[0, perm_x_grid.shape[1], 0, perm_x_grid.shape[2]],
    )
    plt.colorbar(im, ax=ax_map, label="Normalized Log PermX")

    # Logic for unique legend labels
    added_inj_start = False
    added_inj_end = False
    added_prod_start = False
    added_prod_end = False

    for w in range(num_wells):
        is_inj = is_injector[w] > 0.5
        color = "blue" if is_inj else "red"
        marker_end = "^" if is_inj else "v"

        start_label = ""
        end_label = ""

        if is_inj and not added_inj_start:
            start_label = "Injector Start"
            added_inj_start = True
        elif not is_inj and not added_prod_start:
            start_label = "Producer Start"
            added_prod_start = True

        if is_inj and not added_inj_end:
            end_label = "Injector End"
            added_inj_end = True
        elif not is_inj and not added_prod_end:
            end_label = "Producer End"
            added_prod_end = True

        # Plot path (2D: X, Y)
        ax_map.plot(
            history[:, w, 0],
            history[:, w, 1],
            color=color,
            alpha=0.8,
            linestyle="-",
            linewidth=2,
        )

        # Plot start
        ax_map.scatter(
            history[0, w, 0],
            history[0, w, 1],
            color=color,
            marker="o",
            s=60,
            edgecolors="black",
            zorder=5,
            label=start_label if start_label else "",
        )

        # Plot end
        ax_map.scatter(
            history[-1, w, 0],
            history[-1, w, 1],
            color=color,
            marker=marker_end,
            s=180,
            edgecolors="black",
            zorder=6,
            label=end_label if end_label else "",
        )

    ax_map.set_xlabel("X Coordinate")
    ax_map.set_ylabel("Y Coordinate")
    ax_map.set_title(
        f"Well Optimization (2D Projection, {args.optimization_steps} steps)\nBackground: Normalized Log PermX at Depth Z={z_slice}"
    )
    ax_map.legend(bbox_to_anchor=(1.3, 1), loc="upper left")

    # Plot Energy curve
    ax_energy.plot(
        range(len(energy_unnorm)),
        energy_unnorm,
        marker="o",
        linewidth=2,
        color="darkgreen",
    )
    ax_energy.set_xlabel("Optimization Iteration")
    ax_energy.set_ylabel("Total Energy Production (Non-Normalized)")
    ax_energy.set_title(f"Predicted Energy Growth over Differentiable Steps")
    ax_energy.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/well_trajectories_2D_Energy_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved 2D trajectory & energy plot to {plot_path}")


if __name__ == "__main__":
    main()
