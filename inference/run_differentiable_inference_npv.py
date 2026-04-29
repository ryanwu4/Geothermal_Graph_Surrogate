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
from typing import Any
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
from geothermal.economics import compute_real_discount_rate
from geothermal.data import HeteroGraphScaler, build_single_hetero_data
from geothermal.model import HeteroGNNRegressor


SURROGATE_OBJECTIVE_LABELS = {
    "graph_energy_total": "Total Energy Production",
    "graph_energy_rate": "Energy Production Rate",
    "graph_discounted_net_revenue": "Discounted Net Revenue",
}

NPV_OBJECTIVE_LABEL = "NPV (No Facility Terms)"

DEFAULT_NPV_TERMS = {
    # Surrogate and discounting assumptions
    "NOMINAL_DISCOUNT_RATE": 0.09,
    "INFLATION_RATE": 0.02,
    "PLANNING_YEARS": 30,
    # Well CAPEX (no facility terms)
    "WELL_COST_PER_DISTANCE": 2500.0,
    "DEPTH_TO_DISTANCE_SCALE": 24.35,
    # Fixed annual OPEX per well
    "OPEX_WATER_INJECTOR": 200000.0,
    "OPEX_WATER_PRODUCER": 200000.0,
    # Constant-rate pumping OPEX proxy
    "OPEX_ACTIVE_INJECTOR_PER_M3_WATER": 0.22,
    "OPEX_ACTIVE_PRODUCER_PER_M3_WATER": 0.22,
    "INJECTOR_RATE_CONSTANT": 8000.0,
    "PRODUCER_RATE_CONSTANT": -8000.0,
    "ANNUAL_WATER_RATE_SCALE": 365.0,
}


def get_surrogate_label(target: str) -> str:
    if target not in SURROGATE_OBJECTIVE_LABELS:
        valid = sorted(SURROGATE_OBJECTIVE_LABELS.keys())
        raise ValueError(f"Unknown objective target '{target}'. Valid options: {valid}")
    return SURROGATE_OBJECTIVE_LABELS[target]


def load_npv_terms(config: dict) -> dict:
    npv_terms = dict(DEFAULT_NPV_TERMS)

    npv_cfg_path = Path(
        config.get(
            "economics_config",
            config.get("npv_economics_config", "configs/economics.json"),
        )
    )
    if npv_cfg_path.exists():
        with open(npv_cfg_path, "r") as f:
            from_file = json.load(f)
        npv_terms.update(from_file)

    npv_terms.update(config.get("npv_terms", {}))

    required = [
        "PLANNING_YEARS",
        "WELL_COST_PER_DISTANCE",
        "OPEX_WATER_INJECTOR",
        "OPEX_WATER_PRODUCER",
        "OPEX_ACTIVE_INJECTOR_PER_M3_WATER",
        "OPEX_ACTIVE_PRODUCER_PER_M3_WATER",
        "INJECTOR_RATE_CONSTANT",
        "PRODUCER_RATE_CONSTANT",
        "ANNUAL_WATER_RATE_SCALE",
    ]
    missing = [k for k in required if k not in npv_terms]
    if missing:
        raise ValueError(f"Missing required NPV terms: {missing}")

    npv_terms["REAL_DISCOUNT_RATE"] = compute_real_discount_rate(npv_terms)

    return npv_terms


def _resolve_optional_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_geometry_map_config(run_config: dict, npv_terms: dict, config_base_dir: Path) -> dict:
    geometry_cfg_path = _resolve_optional_path(
        run_config.get("geometry_map_config", npv_terms.get("GEOMETRY_MAP_CONFIG", "")),
        config_base_dir,
    )
    if geometry_cfg_path is None:
        return {}
    if not geometry_cfg_path.exists():
        raise FileNotFoundError(f"Geometry map config not found: {geometry_cfg_path}")

    with open(geometry_cfg_path, "r") as f:
        geom_cfg = json.load(f)
    if not isinstance(geom_cfg, dict):
        raise ValueError("Geometry map config must deserialize to a JSON object")

    print(f"Loaded geometry map config from {geometry_cfg_path}")
    return geom_cfg


def _validate_strictly_increasing(name: str, values: torch.Tensor) -> None:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 values")
    diffs = values[1:] - values[:-1]
    if torch.any(diffs <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def build_distance_model(
    npv_terms: dict,
    geometry_cfg: dict,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    model: dict[str, Any] = {}

    profile_cfg = geometry_cfg.get("depth_distance_profile", {})
    z_knots = profile_cfg.get("z_knots", geometry_cfg.get("z_knots"))
    dist_knots = profile_cfg.get("distance_knots", geometry_cfg.get("distance_knots"))

    if z_knots is not None and dist_knots is not None:
        z_knots_t = torch.as_tensor(z_knots, dtype=dtype, device=device)
        dist_knots_t = torch.as_tensor(dist_knots, dtype=dtype, device=device)
        if z_knots_t.ndim != 1 or dist_knots_t.ndim != 1:
            raise ValueError("z_knots and distance_knots must be 1D arrays")
        if z_knots_t.numel() != dist_knots_t.numel():
            raise ValueError("z_knots and distance_knots must have the same length")
        _validate_strictly_increasing("z_knots", z_knots_t)
        if torch.any(dist_knots_t < 0):
            raise ValueError("distance_knots must be non-negative")

        model["depth_mode"] = "piecewise"
        model["z_knots"] = z_knots_t
        model["distance_knots"] = dist_knots_t
    else:
        if "DEPTH_TO_DISTANCE_SCALE" not in npv_terms:
            raise ValueError(
                "Either provide DEPTH_TO_DISTANCE_SCALE in economics terms or "
                "provide z_knots + distance_knots in geometry map config."
            )
        scale = float(npv_terms["DEPTH_TO_DISTANCE_SCALE"])
        if scale <= 0:
            raise ValueError("DEPTH_TO_DISTANCE_SCALE must be > 0")

        model["depth_mode"] = "scalar"
        model["depth_to_distance_scale"] = torch.as_tensor(scale, dtype=dtype, device=device)

    xy_default = float(geometry_cfg.get("xy_scale_default", npv_terms.get("XY_DISTANCE_SCALE_DEFAULT", 1.0)))
    if xy_default <= 0:
        raise ValueError("xy_scale_default must be > 0")
    model["xy_scale_default"] = torch.as_tensor(xy_default, dtype=dtype, device=device)

    xy_cfg = geometry_cfg.get("xy_scale_map")
    if xy_cfg is None:
        top_level_xy_present = all(k in geometry_cfg for k in ["x_knots", "y_knots", "scale_grid"])
        if top_level_xy_present:
            xy_cfg = {
                "x_knots": geometry_cfg.get("x_knots"),
                "y_knots": geometry_cfg.get("y_knots"),
                "scale_grid": geometry_cfg.get("scale_grid"),
            }

    if xy_cfg is not None:
        x_knots = torch.as_tensor(xy_cfg["x_knots"], dtype=dtype, device=device)
        y_knots = torch.as_tensor(xy_cfg["y_knots"], dtype=dtype, device=device)
        scale_grid = torch.as_tensor(xy_cfg["scale_grid"], dtype=dtype, device=device)

        _validate_strictly_increasing("x_knots", x_knots)
        _validate_strictly_increasing("y_knots", y_knots)

        if scale_grid.ndim != 2:
            raise ValueError("scale_grid must be a 2D matrix")
        expected_shape = (y_knots.numel(), x_knots.numel())
        if tuple(scale_grid.shape) != expected_shape:
            raise ValueError(
                f"scale_grid shape {tuple(scale_grid.shape)} must match "
                f"(len(y_knots), len(x_knots))={expected_shape}"
            )
        if torch.any(scale_grid <= 0):
            raise ValueError("scale_grid values must be > 0")

        model["has_xy_scale_map"] = True
        model["x_knots"] = x_knots
        model["y_knots"] = y_knots
        model["scale_grid"] = scale_grid
    else:
        model["has_xy_scale_map"] = False

    print(
        "Distance model: "
        f"depth_mode={model['depth_mode']} | "
        f"xy_map={model['has_xy_scale_map']} | "
        f"xy_default={float(model['xy_scale_default']):.6f}"
    )
    return model


def validate_geometry_model_axes(geometry_model: dict[str, Any], nx: int, ny: int) -> None:
    """Validate that XY geometry map follows optimization axis convention.

    Convention in this script:
    - coords[:, 0] is X and should span [0, nx-1]
    - coords[:, 1] is Y and should span [0, ny-1]
    - scale_grid is indexed as [row_y, col_x]
    """
    if not geometry_model.get("has_xy_scale_map", False):
        return

    x_knots = geometry_model["x_knots"]
    y_knots = geometry_model["y_knots"]

    x_min = float(x_knots[0])
    x_max = float(x_knots[-1])
    y_min = float(y_knots[0])
    y_max = float(y_knots[-1])

    tol = 1.5
    x_ok = abs(x_min - 0.0) <= tol and abs(x_max - float(nx - 1)) <= tol
    y_ok = abs(y_min - 0.0) <= tol and abs(y_max - float(ny - 1)) <= tol
    if x_ok and y_ok:
        return

    looks_swapped = (
        abs(x_min - 0.0) <= tol
        and abs(x_max - float(ny - 1)) <= tol
        and abs(y_min - 0.0) <= tol
        and abs(y_max - float(nx - 1)) <= tol
    )
    if looks_swapped:
        raise ValueError(
            "XY geometry map appears transposed: x_knots match Y range and y_knots match X range. "
            "Expected x_knots to span [0, nx-1] and y_knots to span [0, ny-1]."
        )

    raise ValueError(
        "XY geometry map knot ranges do not match optimization grid bounds. "
        f"Got x=[{x_min:.3f},{x_max:.3f}] and y=[{y_min:.3f},{y_max:.3f}], "
        f"expected approximately x=[0,{nx - 1}] and y=[0,{ny - 1}]."
    )


def torch_interp1d_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    if xp.ndim != 1 or fp.ndim != 1 or xp.numel() != fp.numel():
        raise ValueError("xp and fp must be 1D tensors of equal length")
    if xp.numel() < 2:
        raise ValueError("xp/fp must have at least 2 points for interpolation")

    x_clamped = x.clamp(min=float(xp[0]), max=float(xp[-1]))
    idx = torch.searchsorted(xp, x_clamped, right=True) - 1
    idx = idx.clamp(min=0, max=xp.numel() - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    denom = torch.clamp(x1 - x0, min=torch.finfo(x.dtype).eps)
    t = (x_clamped - x0) / denom
    return y0 + t * (y1 - y0)


def bilinear_interp2d(
    x: torch.Tensor,
    y: torch.Tensor,
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values for bilinear_interp2d must be 2D")

    x_clamped = x.clamp(min=float(x_knots[0]), max=float(x_knots[-1]))
    y_clamped = y.clamp(min=float(y_knots[0]), max=float(y_knots[-1]))

    ix = torch.searchsorted(x_knots, x_clamped, right=True) - 1
    iy = torch.searchsorted(y_knots, y_clamped, right=True) - 1
    ix = ix.clamp(min=0, max=x_knots.numel() - 2)
    iy = iy.clamp(min=0, max=y_knots.numel() - 2)

    x0 = x_knots[ix]
    x1 = x_knots[ix + 1]
    y0 = y_knots[iy]
    y1 = y_knots[iy + 1]

    tx = (x_clamped - x0) / torch.clamp(x1 - x0, min=torch.finfo(x.dtype).eps)
    ty = (y_clamped - y0) / torch.clamp(y1 - y0, min=torch.finfo(y.dtype).eps)

    v00 = values[iy, ix]
    v10 = values[iy, ix + 1]
    v01 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def compute_well_distance(coords: torch.Tensor, geometry_model: dict[str, Any]) -> torch.Tensor:
    z = coords[:, 2]
    if geometry_model["depth_mode"] == "piecewise":
        base_distance = torch_interp1d_linear(
            z,
            geometry_model["z_knots"],
            geometry_model["distance_knots"],
        )
    else:
        base_distance = (z + 1.0) * geometry_model["depth_to_distance_scale"]

    xy_scale = geometry_model["xy_scale_default"].expand_as(z)
    if geometry_model["has_xy_scale_map"]:
        xy_scale = xy_scale * bilinear_interp2d(
            coords[:, 0],
            coords[:, 1],
            geometry_model["x_knots"],
            geometry_model["y_knots"],
            geometry_model["scale_grid"],
        )

    return torch.clamp(base_distance * xy_scale, min=0.0)


def inverse_target_torch(pred_scaled: torch.Tensor, scaler: HeteroGraphScaler) -> torch.Tensor:
    """Differentiable inverse of sklearn StandardScaler for single-dim graph targets."""
    mean = torch.as_tensor(
        float(scaler.target_scaler.mean_[0]),
        dtype=pred_scaled.dtype,
        device=pred_scaled.device,
    )
    scale = torch.as_tensor(
        float(scaler.target_scaler.scale_[0]),
        dtype=pred_scaled.dtype,
        device=pred_scaled.device,
    )
    return pred_scaled * scale + mean


def compute_npv_proxy(
    discounted_revenue: torch.Tensor,
    coords: torch.Tensor,
    is_injector: torch.Tensor,
    npv_terms: dict,
    geometry_model: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute NPV proxy aligned to Julia economics assumptions.

    Included terms:
    - Revenue (from surrogate discounted revenue)
    - CAPEX wells (year 0, undiscounted)
    - OPEX fixed + active-water (discounted)

    Excluded terms by request:
    - CAPEX surface facilities
    - CAPEX flowlines
    - OPEX surface facilities
    """
    dtype = coords.dtype
    device = coords.device
    is_inj_mask = is_injector > 0.5
    is_prod_mask = ~is_inj_mask

    num_inj = is_inj_mask.to(dtype=dtype).sum()
    num_prod = is_prod_mask.to(dtype=dtype).sum()

    well_cost_per_distance = torch.as_tensor(
        float(npv_terms["WELL_COST_PER_DISTANCE"]), dtype=dtype, device=device
    )
    well_distance = compute_well_distance(coords, geometry_model)
    capex_wells = well_distance.sum() * well_cost_per_distance

    opex_water_inj = torch.as_tensor(
        float(npv_terms["OPEX_WATER_INJECTOR"]), dtype=dtype, device=device
    )
    opex_water_prod = torch.as_tensor(
        float(npv_terms["OPEX_WATER_PRODUCER"]), dtype=dtype, device=device
    )
    fixed_opex_annual = num_inj * opex_water_inj + num_prod * opex_water_prod

    inj_rate_const = torch.as_tensor(
        float(npv_terms["INJECTOR_RATE_CONSTANT"]), dtype=dtype, device=device
    )
    prod_rate_const = torch.as_tensor(
        abs(float(npv_terms["PRODUCER_RATE_CONSTANT"])), dtype=dtype, device=device
    )
    annual_scale = torch.as_tensor(
        float(npv_terms["ANNUAL_WATER_RATE_SCALE"]), dtype=dtype, device=device
    )
    opex_active_inj = torch.as_tensor(
        float(npv_terms["OPEX_ACTIVE_INJECTOR_PER_M3_WATER"]), dtype=dtype, device=device
    )
    opex_active_prod = torch.as_tensor(
        float(npv_terms["OPEX_ACTIVE_PRODUCER_PER_M3_WATER"]), dtype=dtype, device=device
    )
    pumping_opex_annual = (
        num_inj * inj_rate_const * annual_scale * opex_active_inj
        + num_prod * prod_rate_const * annual_scale * opex_active_prod
    )

    total_opex_annual = fixed_opex_annual + pumping_opex_annual

    discount_rate = float(npv_terms["REAL_DISCOUNT_RATE"])
    planning_years = int(npv_terms["PLANNING_YEARS"])
    years = torch.arange(1, planning_years + 1, dtype=dtype, device=device)
    discount_vector = (1.0 / (1.0 + discount_rate)) ** years
    discounted_opex = total_opex_annual * discount_vector.sum()

    # NPV proxy with no OPERATED gate by request.
    npv_proxy = -capex_wells + (discounted_revenue - discounted_opex)
    return (
        npv_proxy,
        capex_wells,
        discounted_opex,
        total_opex_annual,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Well Placement Differentiable NPV Optimization (No Facility Terms)"
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
    config_base_dir = args.config.resolve().parent

    surrogate_target = config.get("surrogate_target", "graph_discounted_net_revenue")
    surrogate_label = get_surrogate_label(surrogate_target)
    npv_terms = load_npv_terms(config)
    geometry_cfg = load_geometry_map_config(config, npv_terms, config_base_dir)

    if surrogate_target != "graph_discounted_net_revenue":
        print(
            f"Warning: surrogate_target is '{surrogate_target}'. "
            "NPV proxy assembly assumes surrogate predicts discounted net revenue."
        )

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

        physics_dict["valid_mask"] = torch.tensor(
            valid_mask_cropped, dtype=torch.float32
        )
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
                print(
                    f"Warning: Well at ({x}, {y}) is out of bounds (Grid is {nx}x{ny})"
                )
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
            x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z, porosity, temp0, press0
        )
        vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)

    raw_graph = build_single_hetero_data(
        wells=wells,
        physics_dict=physics_dict,
        full_shape=full_shape,
        target=surrogate_target,
        target_val=0.0,
        vertical_profile=vertical_profiles,
        case_id="inference_custom",
    )

    # Isolate a single case to optimize
    target_graph = scaler.transform_graph(raw_graph)

    # Collate into a batch of size 1
    batch = Batch.from_data_list([target_graph])

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
    coords = batch["well"].pos_xyz.clone().detach().to(device)

    # CRITICAL: We want to optimize the spatial coordinates
    coords.requires_grad = True

    geometry_model = build_distance_model(
        npv_terms=npv_terms,
        geometry_cfg=geometry_cfg,
        dtype=coords.dtype,
        device=device,
    )

    # We use an optimizer to show we can backprop and update the positions
    # (Using Adam to maximize the selected graph-level objective)
    optimizer = optim.Adam([coords], lr=args.learning_rate)

    print("\n--- Starting Well Placement NPV Optimization ---")
    print(f"Initial coordinates:\n{coords.data}")

    # Grid limits from underlying tensor: full_shape is (Z, X, Y)
    z_max, nx, ny = target_graph.physics_context.full_shape
    validate_geometry_model_axes(geometry_model, nx=nx, ny=ny)

    coords_history = []
    npv_history = []
    discounted_revenue_history = []
    capex_history = []
    discounted_opex_history = []
    annual_opex_history = []
    is_injector_tensor = batch["well"].is_injector

    for step in range(args.optimization_steps):
        coords_history.append(coords.clone().detach().cpu().numpy())
        optimizer.zero_grad()

        # Override the batch coordinates dynamically with our differentiable tensor.
        # This channels the gradients through grid_sample cleanly into `coords`.
        batch["well"].pos_xyz = coords

        # Forward Pass! -> physics cropper -> 3D CNN -> HeteroGNN
        predicted_scaled = model(batch).squeeze()
        discounted_revenue = inverse_target_torch(predicted_scaled, scaler)
        (
            npv_proxy,
            capex_wells,
            discounted_opex,
            annual_opex,
        ) = compute_npv_proxy(
            discounted_revenue,
            coords,
            is_injector_tensor,
            npv_terms,
            geometry_model,
        )

        discounted_revenue_history.append(float(discounted_revenue.detach().cpu()))
        capex_history.append(float(capex_wells.detach().cpu()))
        discounted_opex_history.append(float(discounted_opex.detach().cpu()))
        annual_opex_history.append(float(annual_opex.detach().cpu()))
        npv_history.append(float(npv_proxy.detach().cpu()))

        # Maximize NPV proxy by minimizing negative value.
        loss = -npv_proxy

        # Backward Pass!
        loss.backward()

        gradients = coords.grad
        print(
            f"Step {step} | NPV: {npv_proxy.item():.4f} | "
            f"DiscRev: {discounted_revenue.item():.4f} | "
            f"CAPEX_wells: {capex_wells.item():.4f} | "
            f"OPEX_PV: {discounted_opex.item():.4f} | "
            f"MaxGrad: {gradients.abs().max().item():.4f}"
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
        final_scaled = model(batch).squeeze()
        final_discounted_revenue = inverse_target_torch(final_scaled, scaler)
        (
            final_npv_proxy,
            final_capex_wells,
            final_discounted_opex,
            final_annual_opex,
        ) = compute_npv_proxy(
            final_discounted_revenue,
            coords,
            is_injector_tensor,
            npv_terms,
            geometry_model,
        )
        discounted_revenue_history.append(float(final_discounted_revenue.detach().cpu()))
        capex_history.append(float(final_capex_wells.detach().cpu()))
        discounted_opex_history.append(float(final_discounted_opex.detach().cpu()))
        annual_opex_history.append(float(final_annual_opex.detach().cpu()))
        npv_history.append(float(final_npv_proxy.detach().cpu()))

    print("\n--- Optimization Complete ---")
    print(f"Updated Continuous Coordinates:\n{coords.data}")

    # 4. Decode Outputs and Plot Trajectories (2D with background)
    print("\n--- Generating 2D Trajectory, NPV Plots, and Animation ---")

    objective_unnorm = np.array(npv_history, dtype=np.float64)

    history = np.stack(coords_history, axis=0)  # [steps+1, N_wells, 3]
    num_frames, num_wells, _ = history.shape
    removal_depth_threshold = 1.0
    final_depths = history[-1, :, 2]
    removed_well_mask = final_depths < removal_depth_threshold

    is_injector = batch["well"].is_injector.cpu().numpy()

    perm_x_grid = target_graph.physics_context.d["PermX"].cpu().numpy()
    z_slice = perm_x_grid.shape[0] // 2
    background = perm_x_grid[z_slice, :, :].T  # Transpose to (Y, X) for imshow

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
        2,
        figsize=(22, 8),
        gridspec_kw={"width_ratios": [1.2, 1]},
        facecolor=MANIM_BG,
    )
    ax_map = axes[0]
    ax_energy = axes[1]
    _style_ax(ax_map)
    _style_ax(ax_energy)

    im = ax_map.imshow(
        background,
        origin="lower",
        cmap="viridis",
        alpha=0.55,
        extent=[0, perm_x_grid.shape[1], 0, perm_x_grid.shape[2]],
    )
    cbar = plt.colorbar(im, ax=ax_map)
    cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cbar.set_label("Normalized Log PermX", fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE)
    cbar.outline.set_edgecolor(MANIM_WHITE)

    added_inj_trail = False
    added_prod_trail = False
    added_removed_label = False

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

        ax_map.plot(
            history[:, w, 0],
            history[:, w, 1],
            color=color,
            alpha=0.7,
            linestyle="-",
            linewidth=2,
            label=trail_label,
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
        marker_end = "^" if is_inj else "v"
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

        # Cross out wells that end up effectively removed by shallow depth.
        if removed_well_mask[w]:
            removed_label = None
            if not added_removed_label:
                removed_label = f"Removed well (depth < {removal_depth_threshold:g})"
                added_removed_label = True

            ax_map.scatter(
                history[-1, w, 0],
                history[-1, w, 1],
                color=MANIM_WHITE,
                marker="x",
                s=260,
                linewidths=2.6,
                zorder=8,
                label=removed_label,
            )

    ax_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
    ax_map.set_ylabel("Y Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
    ax_map.set_title(
        f"Well Optimization — {args.optimization_steps} Steps\n"
        f"Background: Normalized Log PermX (Z={z_slice})",
        fontsize=TITLE_SIZE,
        color=MANIM_WHITE,
    )
    ax_map.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
    )

    ax_energy.plot(
        range(len(objective_unnorm)),
        objective_unnorm,
        marker="o",
        linewidth=2,
        color=MANIM_BLUE,
        markerfacecolor=MANIM_ORANGE,
        markeredgecolor=MANIM_WHITE,
        markeredgewidth=1.0,
        markersize=7,
    )
    ax_energy.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_energy.set_ylabel(
        f"{NPV_OBJECTIVE_LABEL}",
        fontsize=FONT_SIZE,
        color=MANIM_WHITE,
    )
    ax_energy.set_title(
        f"Predicted {NPV_OBJECTIVE_LABEL} over Differentiable Steps",
        fontsize=TITLE_SIZE,
        color=MANIM_WHITE,
    )
    ax_energy.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)

    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"plots/well_trajectories_2D_npv_proxy_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig)
    print(f"Saved 2D trajectory & NPV plot to {plot_path}")
    if np.any(removed_well_mask):
        removed_ids = np.where(removed_well_mask)[0].tolist()
        print(
            "Crossed out shallow wells on final plot "
            f"(depth < {removal_depth_threshold:g}): indices={removed_ids}"
        )
    print(
        "Final objective breakdown: "
        f"NPV={objective_unnorm[-1]:.4f}, "
        f"DiscountedRevenue={discounted_revenue_history[-1]:.4f}, "
        f"CAPEX_wells={capex_history[-1]:.4f}, "
        f"DiscountedOPEX={discounted_opex_history[-1]:.4f}, "
        f"AnnualOPEX={annual_opex_history[-1]:.4f}, "
        f"SurrogateTarget={surrogate_target} ({surrogate_label})"
    )

    # -------------------------------------------------------------------------
    # 4b. Animated GIF — wells move over time, trails grow behind them
    # -------------------------------------------------------------------------
    import io
    from PIL import Image

    GIF_FPS = 20
    GIF_DPI = 100

    print(f"Building {num_frames} animation frames (PIL fast-path, {GIF_FPS} fps)...")

    # Use the same full-grid extents as the static PNG so proportions match
    grid_xlim = (0, perm_x_grid.shape[1])
    grid_ylim = (0, perm_x_grid.shape[2])

    # Build the figure once; reuse it every frame (Agg backend renders to buffer)
    fig_anim, axes_anim = plt.subplots(
        1,
        2,
        figsize=(22, 8),  # identical size to static PNG
        gridspec_kw={"width_ratios": [1.2, 1]},  # identical ratios to static PNG
        facecolor=MANIM_BG,
    )
    ax_anim_map = axes_anim[0]
    ax_anim_en = axes_anim[1]
    _style_ax(ax_anim_map)
    _style_ax(ax_anim_en)

    # Static permeability background — drawn once, never redrawn
    ax_anim_map.imshow(
        background,
        origin="lower",
        cmap="viridis",
        alpha=0.55,
        extent=[grid_xlim[0], grid_xlim[1], grid_ylim[0], grid_ylim[1]],
    )
    ax_anim_map.set_xlim(*grid_xlim)
    ax_anim_map.set_ylim(*grid_ylim)
    ax_anim_map.set_xlabel("X Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)
    ax_anim_map.set_ylabel("Y Coordinate", fontsize=FONT_SIZE, color=MANIM_WHITE)

    ax_anim_en.set_xlabel(
        "Optimization Iteration", fontsize=FONT_SIZE, color=MANIM_WHITE
    )
    ax_anim_en.set_ylabel(
        f"{NPV_OBJECTIVE_LABEL}",
        fontsize=FONT_SIZE,
        color=MANIM_WHITE,
    )
    ax_anim_en.set_xlim(-0.5, len(objective_unnorm) - 0.5)
    en_margin = (objective_unnorm.max() - objective_unnorm.min()) * 0.12 + 1e-3
    ax_anim_en.set_ylim(
        objective_unnorm.min() - en_margin, objective_unnorm.max() + en_margin
    )
    ax_anim_en.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)

    # Pre-create all mutable artists (updated via set_data each frame)
    trail_lines, dot_artists, removed_artists = [], [], []
    seen_labels: set[str] = set()
    added_removed_legend = False
    for w in range(num_wells):
        is_inj = is_injector[w] > 0.5
        color = MANIM_BLUE if is_inj else MANIM_ORANGE
        label_str = "Injector" if is_inj else "Producer"
        leg_label = label_str if label_str not in seen_labels else None
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
        removed_label = None
        if not added_removed_legend:
            removed_label = f"Removed well (depth < {removal_depth_threshold:g})"
            added_removed_legend = True
        (removed_cross,) = ax_anim_map.plot(
            [],
            [],
            marker="x",
            color=MANIM_WHITE,
            markersize=13,
            markeredgewidth=2.4,
            linestyle="None",
            zorder=8,
            label=removed_label,
        )
        trail_lines.append(trail)
        dot_artists.append(dot)
        removed_artists.append(removed_cross)

    ax_anim_map.legend(
        fontsize=LEGEND_SIZE,
        facecolor="#111111",
        edgecolor=MANIM_GREY,
        labelcolor=MANIM_WHITE,
        loc="upper right",
    )

    (en_line,) = ax_anim_en.plot(
        [],
        [],
        color=MANIM_BLUE,
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor=MANIM_ORANGE,
        markeredgecolor=MANIM_WHITE,
        markeredgewidth=1.0,
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
    title_obj = ax_anim_map.set_title("", fontsize=TITLE_SIZE, color=MANIM_WHITE)
    ax_anim_en.set_title(
        f"Predicted {NPV_OBJECTIVE_LABEL}", fontsize=TITLE_SIZE, color=MANIM_WHITE
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Fast manual render loop: draw each frame → Agg buffer → PIL Image
    # ------------------------------------------------------------------
    pil_frames: list[Image.Image] = []
    canvas = fig_anim.canvas
    canvas.draw()  # initial draw to populate the renderer

    for frame in range(num_frames):
        # Update trail + dot per well
        for w in range(num_wells):
            xs = history[: frame + 1, w, 0]
            ys = history[: frame + 1, w, 1]
            trail_lines[w].set_data(xs, ys)
            dot_artists[w].set_data([xs[-1]], [ys[-1]])

            # Once a well ever gets shallow enough, mark it as effectively removed.
            removed_now = np.any(history[: frame + 1, w, 2] < removal_depth_threshold)
            if removed_now:
                removed_artists[w].set_data([xs[-1]], [ys[-1]])
            else:
                removed_artists[w].set_data([], [])

        # Energy panel
        en_frame = min(frame, len(objective_unnorm) - 1)
        en_line.set_data(range(en_frame + 1), objective_unnorm[: en_frame + 1])
        en_dot.set_offsets([[en_frame, objective_unnorm[en_frame]]])

        title_obj.set_text(
            f"Well Placement Optimization — Step {frame} / {num_frames - 1}\n"
            f"Background: Normalized Log PermX (Z={z_slice})"
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

    gif_path = f"plots/well_optimization_npv_proxy_{timestamp}.gif"
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
