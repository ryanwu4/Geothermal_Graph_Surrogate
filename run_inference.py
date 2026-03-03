#!/usr/bin/env python3
"""Run inference on a custom JSON configuration using a trained GNN.

Usage:
    python run_inference.py --config my_wells.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from compile_minimal_geothermal_h5 import (
    extract_well_data,
    build_wells_table,
    extract_vertical_profiles,
)
from build_geology_graph import generate_geology_edges
from geothermal.data import HeteroGraphScaler, build_single_hetero_data
from geothermal.model import HeteroGNNRegressor

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict total field energy from a well layout JSON.")
    parser.add_argument("--config", type=Path, required=True, help="Path to input JSON configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    checkpoint_path = config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Missing 'checkpoint' path in JSON configuration")
    scaler_path = config.get("scaler_path")
    if not scaler_path:
        raise ValueError("Missing 'scaler_path' in JSON configuration")

    device_str = config.get("device", "cpu")
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str.lower() == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using inference device: {device}")

    geology_path = Path(config["geology_h5_file"])
    if not geology_path.exists():
        raise FileNotFoundError(f"Geology H5 file not found: {geology_path}")

    with h5py.File(geology_path, "r") as src:
        # Determine the Z cutoff using Temperature0 < -900 logic
        print("Reading geology properties and computing depth horizon cutoff...")
        temp0_full = src["Input/Temperature0"][:]
        z_max, nx, ny = temp0_full.shape
        cutoff_z = z_max
        for z in range(z_max):
            slice_z = temp0_full[z, :, :]
            inactive = np.sum(slice_z <= -900)
            if inactive / slice_z.size > 0.95:
                cutoff_z = z
                break
        
        print(f"Computed depth horizon cutoff at Z={cutoff_z} (out of {z_max})")

        # Create virtual grids for wells mimicking the simulator state
        is_well = np.zeros((z_max, nx, ny), dtype=np.int32)
        inj_rate = np.zeros((z_max, nx, ny), dtype=np.float32)

        for w in config["wells"]:
            x, y = w["x"], w["y"]
            w_type = w.get("type", "injector").lower()
            
            # Ensure wells do not go past the bounds
            if x < 0 or x >= nx or y < 0 or y >= ny:
                print(f"Warning: Well at ({x}, {y}) is out of bounds (Grid is {nx}x{ny})")
                continue

            # Perforate down to the cutoff, masking out any inactive fault cells
            for z in range(cutoff_z):
                if temp0_full[z, x, y] <= -900:
                    is_well[z, x, y] = -999
                    inj_rate[z, x, y] = -999
                else:
                    is_well[z, x, y] = 1
                    if w_type == "injector":
                        inj_rate[z, x, y] = 8000.0  # standard magnitude
                    else:
                        inj_rate[z, x, y] = -8000.0
                
        # Use shared extractor methods
        print("Extracting per-well base properties at perforated layers...")
        (
            x_idx, y_idx, depth, inj,
            perm_x, perm_y, perm_z, porosity,
            temp0, press0, depth_centroid,
        ) = extract_well_data(is_well, inj_rate, src)
        
        wells = build_wells_table(
            x_idx, y_idx, depth, inj,
            perm_x, perm_y, perm_z, porosity, temp0, press0
        )
        
        vertical_profile = extract_vertical_profiles(is_well, x_idx, y_idx, src)

        perm_x_full = src["Input/PermX"][:]
        perm_y_full = src["Input/PermY"][:]
        perm_z_full = src["Input/PermZ"][:]
        porosity_full = src["Input/Porosity"][:]
        press0_full = src["Input/Pressure0"][:]
        perm_avg_full = (perm_x_full + perm_y_full + perm_z_full) / 3.0

        well_coords = np.stack([x_idx, y_idx, depth_centroid], axis=1)

        print("Executing A* graph generation for connectivity...")
        # Add a dummy batch parameter or just use generate_geology_edges
        edge_index, edge_attr = generate_geology_edges(
            perm_avg_full,
            porosity_full,
            temp0_full,
            press0_full,
            inj > 0,
            well_coords,
            k_neighbors=2,
        )

    # Assembly
    data = build_single_hetero_data(
        wells=wells,
        vertical_profile=vertical_profile,
        geo_idx=edge_index,
        geo_attr=edge_attr,
        target="graph_energy_total",
        target_val=0.0,
        case_id="inference_custom"
    )

    print(f"Loading scaler from {scaler_path} ...")
    with open(scaler_path, "rb") as f:
        scaler: HeteroGraphScaler = pickle.load(f)

    # Need scaling logic. Using our instance scale method
    print("Transforming graph via scaler...")
    data_scaled = scaler.transform_graph(data)

    print(f"Loading model checkpoint from {checkpoint_path} ...")
    model = HeteroGNNRegressor.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model = model.to(device)

    # Inference requires placing it into a DataLoader batch equivalent
    batch = Batch.from_data_list([data_scaled]).to(device)
    
    print(f"Executing forward pass on {device}...")
    with torch.no_grad():
        preds_scaled = model(batch).cpu().numpy()

    preds = scaler.inverse_targets(preds_scaled)
    predicted_val = float(preds.flat[0])

    print("\n" + "=" * 55)
    print(f"    Predicted Total Energy Production: {predicted_val:,.2f}")
    if "actual_total_energy" in config:
        err = 100 * abs(predicted_val - config["actual_total_energy"]) / max(1, config["actual_total_energy"])
        print(f"    Ground Truth Total Energy:       {config['actual_total_energy']:,.2f}")
        print(f"    Error:                           {err:.2f}%")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
