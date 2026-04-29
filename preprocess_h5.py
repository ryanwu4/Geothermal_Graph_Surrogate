#!/usr/bin/env python3
"""
Lightweight Data Preprocessing Script for 3D CNN Physics Slab Integration.

This script processes original H5 simulations to create an ML-ready dataset:
1. Extracts first-timestep static properties: PermX, PermY, PermZ, Porosity, Temp0, Press0.
2. Identifies a valid rock mask (filtering out sentinel values like -999).
3. Applies Z-axis cropping: cuts off the domain at the deepest z-layer where 95% of cells are invalid.
4. Computes global min/max normalization stats across all files (or uses an existing config).
5. Log-transforms permeability, min-max scales all properties to [0, 1].
6. Clamps out-of-bounds/invalid cells strictly to 0.0.
7. Saves the processed 3D tensors and the well tables into a new H5 file,
   and saves the normalization parameters to a JSON file.
"""

import argparse
import json
import logging
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from geothermal.economics import (
    compute_discounted_net_energy_revenue,
    compute_real_discount_rate,
)

# Try to import well extraction utilities from existing codebase
try:
    from compile_minimal_geothermal_h5 import (
        extract_well_data,
        build_wells_table,
        extract_wept_for_wells,
        extract_well_tp_profiles,
        extract_vertical_profiles,
    )
except ImportError:
    print("Warning: Could not import well utilities from compile_minimal_geothermal_h5.py. Please ensure it is in the same directory.")
    exit(1)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


PROPERTIES = ["PermX", "PermY", "PermZ", "Porosity", "Temperature0", "Pressure0"]
PERM_PROPS = {"PermX", "PermY", "PermZ"}
WEPT_NULL_VALUE = np.float32(-999.0)

DEFAULT_ECONOMICS_CONFIG = Path(__file__).parent / "configs" / "economics.json"


def load_economics_config(path: Path) -> dict:
    with path.open("r") as f:
        economics = json.load(f)

    required = ["ENERGY_PRICE"]
    missing = [k for k in required if k not in economics]
    if missing:
        raise ValueError(
            f"Economics config is missing required keys: {missing}. "
            f"Found keys: {sorted(economics.keys())}"
        )

    if not (
        ("NOMINAL_DISCOUNT_RATE" in economics and "INFLATION_RATE" in economics)
        or "REAL_DISCOUNT_RATE" in economics
    ):
        raise ValueError(
            "Economics config must define NOMINAL_DISCOUNT_RATE and INFLATION_RATE "
            "or REAL_DISCOUNT_RATE."
        )

    economics["REAL_DISCOUNT_RATE"] = compute_real_discount_rate(economics)

    return economics


def get_valid_mask(src_group: h5py.Group) -> np.ndarray:
    """Create a boolean mask of valid cells (ignoring sentinel values like -999)."""
    # Usually -999 is the sentinel for out-of-bounds.
    # Porosity is typically > 0 for valid rock. We'll use Porosity > 0 and PermX > -900 as a proxy.
    poro = src_group["Input/Porosity"][:]
    perm_x = src_group["Input/PermX"][:]
    # Valid if porosity is physically meaningful and perm isn't a sentinel
    valid = (poro > 1e-5) & (perm_x > -900.0)
    return valid


def find_z_cutoff(mask: np.ndarray, invalid_threshold: float = 0.95) -> int:
    """Find the Z-index where the layer is >= invalid_threshold% invalid cells.
    Assumes Z is the first dimension. Ignores initial invalid layers (overburden).
    """
    z_layers = mask.shape[0]
    total_cells_per_layer = mask.shape[1] * mask.shape[2]
    
    found_reservoir = False
    
    for z in range(z_layers):
        invalid_cells = total_cells_per_layer - np.sum(mask[z])
        invalid_ratio = invalid_cells / total_cells_per_layer
        
        if invalid_ratio < invalid_threshold:
            found_reservoir = True
            
        if found_reservoir and invalid_ratio >= invalid_threshold:
            return z
            
    return z_layers


def compute_chunk_stats(filepath: Path) -> dict:
    """Compute min/max for all valid cells in a single H5 file."""
    stats = {p: {"min": float("inf"), "max": float("-inf")} for p in PROPERTIES}
    
    try:
        with h5py.File(filepath, "r") as f:
            valid_mask = get_valid_mask(f)
            z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
            
            valid_mask_cropped = valid_mask[:z_cutoff]
            if not np.any(valid_mask_cropped):
                return {"status": "empty", "filepath": str(filepath)}
            
            for prop in PROPERTIES:
                data = f[f"Input/{prop}"][:z_cutoff]
                valid_data = data[valid_mask_cropped]
                
                if prop in PERM_PROPS:
                    # Log transform permeability for stats calculation
                    # Add small epsilon to avoid log(0)
                    valid_data = np.log10(np.maximum(valid_data, 1e-15))
                
                if len(valid_data) > 0:
                    stats[prop]["min"] = float(np.min(valid_data))
                    stats[prop]["max"] = float(np.max(valid_data))
                    
        return {"status": "success", "stats": stats, "filepath": str(filepath)}
    except Exception as e:
        return {"status": "error", "error": str(e), "filepath": str(filepath)}


def process_file_and_save(
    filepath: Path,
    output_h5_path: Path,
    norm_config: dict,
    economics: dict,
) -> dict:
    """Read a file, normalize it using the config, completely crop it, and save to output H5."""
    try:
        with h5py.File(filepath, "r") as src:
            is_well = src["Input/IsWell"][:]
            inj_rate = src["Input/InjRate"][:]
            
            (
                x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z, 
                porosity, temp0, press0, depth_centroid
            ) = extract_well_data(is_well, inj_rate, src)
            
            wells = build_wells_table(
                x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z, 
                porosity, temp0, press0
            )

            energy_total = src["Output/FieldEnergyProductionTotal"][...].astype(np.float32)
            energy_rate = src["Output/FieldEnergyProductionRate"][...].astype(np.float32)
            energy_inj_rate = src["Output/FieldEnergyInjectionRate"][...].astype(np.float32)
            discounted_net_revenue = compute_discounted_net_energy_revenue(
                energy_rate,
                energy_inj_rate,
                economics,
            )

            # WEPT may be missing in some wrapper-generated v2.5 files.
            # In that case, keep the case and write a null sentinel target.
            if "Output/WEPT" in src:
                well_wept = extract_wept_for_wells(src, x_idx, y_idx, depth)
            else:
                logging.warning(
                    "Output/WEPT missing in %s. Using null WEPT sentinel (%s).",
                    filepath,
                    WEPT_NULL_VALUE,
                )
                well_wept = np.full((len(wells), 1), WEPT_NULL_VALUE, dtype=np.float32)

            well_tp = extract_well_tp_profiles(src, is_well, x_idx, y_idx)
            vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)

            valid_mask = get_valid_mask(src)
            z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
            valid_mask_cropped = valid_mask[:z_cutoff]
            
            # Prepare normalized tensors
            tensor_dict = {}
            for prop in PROPERTIES:
                data = src[f"Input/{prop}"][:z_cutoff].astype(np.float32)
                
                if prop in PERM_PROPS:
                    data = np.log10(np.maximum(data, 1e-15))
                
                p_min = norm_config[prop]["min"]
                p_max = norm_config[prop]["max"]
                
                # Min-max scale
                if p_max > p_min:
                    normalized = (data - p_min) / (p_max - p_min)
                else:
                    normalized = np.zeros_like(data)
                
                # Clamp out of bounds and mask invalid to 0.0 strictly
                normalized = np.clip(normalized, 0.0, 1.0)
                normalized[~valid_mask_cropped] = 0.0
                
                tensor_dict[prop] = normalized

            # Write to the destination H5
            case_id = filepath.stem
            
            # Safe parallel write logic: we return data to master, OR we write sequentially
            # To avoid locking issues in HDF5 multiprocessing, we'll return the arrays 
            # and let the master process write them. Memory might be tight, so let's
            # return the dict.
            return {
                "status": "success",
                "case_id": case_id,
                "wells": wells,
                "energy_total": energy_total,
                "energy_rate": energy_rate,
                "energy_inj_rate": energy_inj_rate,
                "discounted_net_revenue": np.float32(discounted_net_revenue),
                "well_wept": well_wept,
                "well_tp": well_tp,
                "vertical_profiles": vertical_profiles,
                "tensors": tensor_dict,
                "valid_mask": valid_mask_cropped.astype(np.float32),
                "z_cutoff": int(z_cutoff)
            }

    except Exception as e:
        return {"status": "error", "error": str(e), "filepath": str(filepath)}


def main():
    parser = argparse.ArgumentParser(description="Preprocess Geothermal H5 into lightweight ML tensors.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with original .h5 files")
    parser.add_argument("--output-h5", type=Path, required=True, help="Path for output normalized .h5 dataset")
    parser.add_argument("--norm-config", type=Path, default=Path("norm_config.json"), help="Path to save/load normalization JSON file")
    parser.add_argument(
        "--economics-config",
        type=Path,
        default=DEFAULT_ECONOMICS_CONFIG,
        help="Path to economics JSON used to compute discounted net revenue target",
    )
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()), help="Number of workers")
    parser.add_argument("--compute-only", action="store_true", help="Only compute and save norm_config, don't build H5")
    
    args = parser.parse_args()
    
    h5_files = sorted(list(args.input_dir.glob("*.h5")))
    if not h5_files:
        logging.error(f"No H5 files found in {args.input_dir}")
        return

    if not args.economics_config.exists():
        logging.error(f"Economics config not found: {args.economics_config}")
        return

    economics = load_economics_config(args.economics_config)

    # Pass 1: Compute or Load Global Normalization Statistics
    if args.norm_config.exists() and not args.compute_only:
        logging.info(f"Loading normalization configuration from {args.norm_config}")
        with open(args.norm_config, "r") as f:
            global_stats = json.load(f)
    else:
        logging.info("Computing global statistics over all files for normalization (Pass 1/2)...")
        global_stats = {p: {"min": float("inf"), "max": float("-inf")} for p in PROPERTIES}
        
        with mp.Pool(args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(compute_chunk_stats, h5_files), total=len(h5_files)))
            
        for res in results:
            if res["status"] == "success":
                for p in PROPERTIES:
                    global_stats[p]["min"] = min(global_stats[p]["min"], res["stats"][p]["min"])
                    global_stats[p]["max"] = max(global_stats[p]["max"], res["stats"][p]["max"])
            elif res["status"] == "error":
                logging.warning(f"Error computing stats for {res.get('filepath')}: {res.get('error')}")
                
        # Save config
        with open(args.norm_config, "w") as f:
            json.dump(global_stats, f, indent=4)
        logging.info(f"Saved normalization configuration to {args.norm_config}")
        
    if args.compute_only:
        logging.info("Compute-only flag active. Exiting.")
        return

    # Pass 2: Process and Write Output
    logging.info(f"Building aggregated lightweight ML dataset: {args.output_h5} (Pass 2/2)...")
    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(args.output_h5, "w") as out_f:
        out_f.attrs["description"] = "Lightweight ML dataset with normalized physics slabs."
        for p in PROPERTIES:
            out_f.attrs[f"norm_{p}_min"] = global_stats[p]["min"]
            out_f.attrs[f"norm_{p}_max"] = global_stats[p]["max"]
        out_f.attrs["target_graph_discounted_net_revenue_energy_price_kwh"] = float(
            economics["ENERGY_PRICE"]
        )
        real_discount_rate = compute_real_discount_rate(economics)
        out_f.attrs["target_graph_discounted_net_revenue_real_discount_rate"] = float(
            real_discount_rate
        )
        if "NOMINAL_DISCOUNT_RATE" in economics:
            out_f.attrs["target_graph_discounted_net_revenue_nominal_discount_rate"] = float(
                economics["NOMINAL_DISCOUNT_RATE"]
            )
        if "INFLATION_RATE" in economics:
            out_f.attrs["target_graph_discounted_net_revenue_inflation_rate"] = float(
                economics["INFLATION_RATE"]
            )
        out_f.attrs["target_graph_discounted_net_revenue_formula"] = (
            "sum_t ((FEPR_t - FEIR_t) * (ENERGY_PRICE/3600) * (1/(1+r))^t), t starts at 1"
        )
        out_f.attrs["target_graph_discounted_net_revenue_includes_capex"] = 0
        out_f.attrs["target_graph_discounted_net_revenue_includes_opex"] = 0

        from functools import partial
        process_fn = partial(
            process_file_and_save,
            output_h5_path=args.output_h5,
            norm_config=global_stats,
            economics=economics,
        )
        
        with mp.Pool(args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_fn, h5_files), total=len(h5_files)):
                if res["status"] == "success":
                    grp = out_f.create_group(res["case_id"])
                    
                    # Store continuous target and well metadata
                    grp.create_dataset("field_energy_production_total", data=res["energy_total"])
                    grp.create_dataset("field_energy_production_rate", data=res["energy_rate"])
                    grp.create_dataset("field_energy_injection_rate", data=res["energy_inj_rate"])
                    grp.create_dataset(
                        "field_discounted_net_revenue",
                        data=res["discounted_net_revenue"],
                    )
                    grp.create_dataset("well_wept", data=res["well_wept"], compression="gzip")
                    grp.create_dataset("well_tp_profiles", data=res["well_tp"], compression="gzip")
                    grp.create_dataset("well_vertical_profile", data=res["vertical_profiles"], compression="gzip")
                    
                    grp.create_dataset("wells", data=res["wells"], compression="gzip")
                    grp.attrs["z_cutoff"] = res["z_cutoff"]
                    
                    # Store tensor channels
                    tensor_grp = grp.create_group("physics_tensors")
                    for prop in PROPERTIES:
                        tensor_grp.create_dataset(
                            prop, 
                            data=res["tensors"][prop], 
                            compression="gzip", 
                            compression_opts=4
                        )
                    tensor_grp.create_dataset(
                        "valid_mask", 
                        data=res["valid_mask"], 
                        compression="gzip", 
                        compression_opts=4
                    )
                else:
                    logging.warning(f"Error processing {res.get('filepath')}: {res.get('error')}")

    logging.info(f"Successfully generated lightweight ML dataset at {args.output_h5}")

if __name__ == "__main__":
    main()
