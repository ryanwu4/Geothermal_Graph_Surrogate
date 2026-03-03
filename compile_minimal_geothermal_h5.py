#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial

import h5py
import numpy as np
from tqdm import tqdm

from build_geology_graph import generate_geology_edges


def extract_well_data(
    is_well: np.ndarray,
    inj_rate: np.ndarray,
    src: h5py.File,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract per-well data at the deepest perforated layer.

    Returns:
        x_idx, y_idx, depth (deepest Z), inj_rate,
        perm_x, perm_y, perm_z, porosity, temp0, press0,
        depth_centroid (perm-weighted centroid Z index)
    """
    if is_well.shape != inj_rate.shape:
        raise ValueError(
            f"Shape mismatch: IsWell {is_well.shape} vs InjRate {inj_rate.shape}"
        )

    if is_well.ndim != 3:
        raise ValueError(f"Expected 3D well grids [z, x, y], got shape {is_well.shape}")

    well_mask = is_well == 1
    has_well_xy = np.any(well_mask, axis=0)

    x_idx, y_idx = np.where(has_well_xy)
    if x_idx.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    z_count = is_well.shape[0]
    z_indices = np.arange(z_count, dtype=np.int32)[:, None, None]

    deepest_z_all = np.where(well_mask, z_indices, -1).max(axis=0)
    depth = deepest_z_all[x_idx, y_idx].astype(np.int32, copy=False)

    inj = inj_rate[depth, x_idx, y_idx].astype(np.float32, copy=False)

    unique_z = np.unique(depth)

    perm_x = np.zeros_like(inj)
    perm_y = np.zeros_like(inj)
    perm_z = np.zeros_like(inj)
    porosity = np.zeros_like(inj)
    temp0 = np.zeros_like(inj)
    press0 = np.zeros_like(inj)

    for z in unique_z:
        mask = depth == z
        x_z = x_idx[mask]
        y_z = y_idx[mask]

        px_slice = src["Input/PermX"][z, :, :]
        py_slice = src["Input/PermY"][z, :, :]
        pz_slice = src["Input/PermZ"][z, :, :]
        por_slice = src["Input/Porosity"][z, :, :]
        t0_slice = src["Input/Temperature0"][z, :, :]
        p0_slice = src["Input/Pressure0"][z, :, :]

        perm_x[mask] = px_slice[x_z, y_z]
        perm_y[mask] = py_slice[x_z, y_z]
        perm_z[mask] = pz_slice[x_z, y_z]
        porosity[mask] = por_slice[x_z, y_z]
        temp0[mask] = t0_slice[x_z, y_z]
        press0[mask] = p0_slice[x_z, y_z]

    # Compute permeability-weighted centroid depth for each well.
    # This is the depth at which flow effectively concentrates across the
    # perforated interval, and serves as a more physical A* start point
    # than the absolute bottom of the well.
    perm_avg_grid = (
        src["Input/PermX"][:] + src["Input/PermY"][:] + src["Input/PermZ"][:]
    ) / 3.0
    depth_centroid = np.zeros(x_idx.size, dtype=np.int32)
    for i in range(x_idx.size):
        xi, yi = x_idx[i], y_idx[i]
        z_perf = np.where(well_mask[:, xi, yi])[0]
        if len(z_perf) <= 1:
            depth_centroid[i] = depth[i]
        else:
            perms = perm_avg_grid[z_perf, xi, yi].astype(np.float64)
            perms = np.maximum(perms, 1e-30)  # avoid zero weights
            centroid_z = np.average(z_perf.astype(np.float64), weights=perms)
            # Snap to nearest valid perforated layer
            depth_centroid[i] = z_perf[np.argmin(np.abs(z_perf - centroid_z))]

    return (
        x_idx.astype(np.int32),
        y_idx.astype(np.int32),
        depth,
        inj,
        perm_x,
        perm_y,
        perm_z,
        porosity,
        temp0,
        press0,
        depth_centroid,
    )


def build_wells_table(
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    depth: np.ndarray,
    inj: np.ndarray,
    perm_x: np.ndarray,
    perm_y: np.ndarray,
    perm_z: np.ndarray,
    porosity: np.ndarray,
    temp0: np.ndarray,
    press0: np.ndarray,
) -> np.ndarray:
    wells_dtype = np.dtype(
        [
            ("x", np.int32),
            ("y", np.int32),
            ("depth", np.int32),
            ("inj_rate", np.float32),
            ("perm_x", np.float32),
            ("perm_y", np.float32),
            ("perm_z", np.float32),
            ("porosity", np.float32),
            ("temp0", np.float32),
            ("press0", np.float32),
        ]
    )
    wells = np.empty(x_idx.shape[0], dtype=wells_dtype)
    wells["x"] = x_idx
    wells["y"] = y_idx
    wells["depth"] = depth
    wells["inj_rate"] = inj
    wells["perm_x"] = perm_x
    wells["perm_y"] = perm_y
    wells["perm_z"] = perm_z
    wells["porosity"] = porosity
    wells["temp0"] = temp0
    wells["press0"] = press0
    return wells


def extract_vertical_profiles(
    is_well: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    src: h5py.File,
) -> np.ndarray:
    """Compute vertical profile summary statistics for each well.

    For each well at (x, y), reads all perforated Z-layers and computes
    mean, min, max, std for 6 properties (perm_x, perm_y, perm_z,
    porosity, temp0, press0) plus n_layers.

    Returns:
        Array of shape [N_wells, 25] (6 props * 4 stats + 1 n_layers).
    """
    n_wells = x_idx.size
    # 6 properties * 4 stats (mean, min, max, std) + 1 n_layers = 25
    N_PROFILE_FEATURES = 25

    if n_wells == 0:
        return np.empty((0, N_PROFILE_FEATURES), dtype=np.float32)

    well_mask = is_well == 1

    prop_names = [
        "Input/PermX",
        "Input/PermY",
        "Input/PermZ",
        "Input/Porosity",
        "Input/Temperature0",
        "Input/Pressure0",
    ]
    # Read full grids once (they're already in memory for A* anyway)
    grids = [src[name][:] for name in prop_names]

    profiles = np.zeros((n_wells, N_PROFILE_FEATURES), dtype=np.float32)

    for i in range(n_wells):
        xi, yi = x_idx[i], y_idx[i]
        z_perf = np.where(well_mask[:, xi, yi])[0]
        n_layers = len(z_perf)

        col = 0
        for grid in grids:
            vals = grid[z_perf, xi, yi].astype(np.float64)
            # Filter sentinel -999 values for thermodynamic properties
            valid = vals[vals > -900]
            if len(valid) == 0:
                valid = np.array([0.0])
            profiles[i, col] = np.mean(valid)
            profiles[i, col + 1] = np.min(valid)
            profiles[i, col + 2] = np.max(valid)
            profiles[i, col + 3] = np.std(valid) if len(valid) > 1 else 0.0
            col += 4

        profiles[i, col] = float(n_layers)

    return profiles


def extract_wept_for_wells(
    src: h5py.File,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    wept_ds = src["Output/WEPT"]
    if wept_ds.ndim != 4:
        raise ValueError(f"Expected WEPT shape [time, z, x, y], got {wept_ds.shape}")

    t_steps = wept_ds.shape[0]
    if x_idx.size == 0:
        return np.empty((0, t_steps), dtype=np.float32)

    # Read only the specific Z-slices we need for all timesteps
    unique_z = np.unique(depth)
    wept_series = np.zeros((x_idx.size, t_steps), dtype=np.float32)

    for z in unique_z:
        mask = depth == z
        x_z = x_idx[mask]
        y_z = y_idx[mask]

        # Read the entire Z-slice for all timesteps
        # Shape: [time, x, y]
        wept_slice = wept_ds[:, z, :, :]

        # Extract the specific points
        wept_series[mask] = wept_slice[:, x_z, y_z].T

    return wept_series


def extract_well_tp_profiles(
    src: h5py.File,
    is_well: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> np.ndarray:
    """Extract per-well temperature and pressure profile statistics at each timestep.

    For each well at (x, y), reads only the perforated Z-layers from
    Output/Temperature and Output/Pressure at every timestep and computes
    mean, min, max over the vertical profile.

    Reads are targeted to individual well columns (all timesteps at once)
    to minimize I/O over NFS.

    Returns:
        Array of shape [N_wells, N_timesteps, 6] where the last axis is:
        [mean_T, min_T, max_T, mean_P, min_P, max_P]
    """
    temp_ds = src["Output/Temperature"]  # (T, Z, X, Y)
    press_ds = src["Output/Pressure"]  # (T, Z, X, Y)
    n_timesteps = temp_ds.shape[0]
    n_wells = x_idx.size
    N_STATS = 6  # mean/min/max for T and P

    if n_wells == 0:
        return np.empty((0, n_timesteps, N_STATS), dtype=np.float32)

    well_mask = is_well == 1
    profiles = np.zeros((n_wells, n_timesteps, N_STATS), dtype=np.float32)

    for i in range(n_wells):
        xi, yi = int(x_idx[i]), int(y_idx[i])
        z_perf = np.where(well_mask[:, xi, yi])[0]

        # Read only this well's column: shape (T, len(z_perf))
        # HDF5 fancy indexing: read all timesteps for specific (z, x, y)
        t_col = temp_ds[:, z_perf, xi, yi].astype(np.float64)  # (T, n_layers)
        p_col = press_ds[:, z_perf, xi, yi].astype(np.float64)  # (T, n_layers)

        for t in range(n_timesteps):
            t_vals = t_col[t]
            p_vals = p_col[t]

            # Filter sentinel -999 values
            t_valid = t_vals[t_vals > -900]
            p_valid = p_vals[p_vals > -900]
            if len(t_valid) == 0:
                t_valid = np.array([0.0])
            if len(p_valid) == 0:
                p_valid = np.array([0.0])

            profiles[i, t, 0] = np.mean(t_valid)
            profiles[i, t, 1] = np.min(t_valid)
            profiles[i, t, 2] = np.max(t_valid)
            profiles[i, t, 3] = np.mean(p_valid)
            profiles[i, t, 4] = np.min(p_valid)
            profiles[i, t, 5] = np.max(p_valid)

    return profiles


def _process_single_file(source_path: Path) -> dict | None:
    """Process a single HDF5 file and return the extracted data as a dictionary."""
    try:
        with h5py.File(source_path, "r") as src:
            required_paths = [
                "Input/IsWell",
                "Input/InjRate",
                "Input/PermX",
                "Input/PermY",
                "Input/PermZ",
                "Input/Porosity",
                "Input/Temperature0",
                "Input/Pressure0",
                "Output/FieldEnergyInjectionRate",
                "Output/FieldEnergyProductionRate",
                "Output/FieldEnergyProductionTotal",
                "Output/WEPT",
                "Output/Temperature",
                "Output/Pressure",
            ]
            if not all(path in src for path in required_paths):
                return {
                    "status": "skipped",
                    "name": source_path.name,
                    "reason": "missing required datasets",
                }

            is_well = src["Input/IsWell"][...]
            inj_rate = src["Input/InjRate"][...]

            # Full 3D arrays for A* (read once for the file)
            perm_x_full = src["Input/PermX"][:]
            perm_y_full = src["Input/PermY"][:]
            perm_z_full = src["Input/PermZ"][:]
            porosity_full = src["Input/Porosity"][:]
            temp0_full = src["Input/Temperature0"][:]
            press0_full = src["Input/Pressure0"][:]

            # Calculate an isotropic or average permeability for A* routing
            perm_avg_full = (perm_x_full + perm_y_full + perm_z_full) / 3.0

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

            # Extract vertical profile summary statistics
            vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)

            # Use perm-weighted centroid as A* start/end points:
            # fluid enters/exits preferentially through the most permeable
            # layers, so the centroid is a better representative depth than
            # the absolute bottom of the well.
            well_coords = np.stack(
                [x_idx, y_idx, depth_centroid], axis=1
            )  # [X, Y, Z] used by generate_geology_edges
            edge_index, edge_attr = generate_geology_edges(
                perm_avg_full,
                porosity_full,
                temp0_full,
                press0_full,
                inj > 0,
                well_coords,
                k_neighbors=2,
            )

            energy_inj = src["Output/FieldEnergyInjectionRate"][...].astype(
                np.float32, copy=False
            )
            energy_prod = src["Output/FieldEnergyProductionRate"][...].astype(
                np.float32, copy=False
            )
            energy_total = src["Output/FieldEnergyProductionTotal"][...].astype(
                np.float32, copy=False
            )
            well_wept = extract_wept_for_wells(src, x_idx, y_idx, depth)
            well_tp_profiles = extract_well_tp_profiles(src, is_well, x_idx, y_idx)

            return {
                "status": "success",
                "name": source_path.name,
                "stem": source_path.stem,
                "source_file": str(source_path),
                "wells": wells,
                "well_wept": well_wept,
                "well_tp_profiles": well_tp_profiles,
                "well_vertical_profile": vertical_profiles,
                "geology_edge_index": edge_index,
                "geology_edge_attr": edge_attr,
                "energy_inj": energy_inj,
                "energy_prod": energy_prod,
                "energy_total": energy_total,
                "well_count": int(x_idx.size),
            }
    except Exception as e:
        return {"status": "error", "name": source_path.name, "reason": str(e)}


def compile_dataset(
    input_dir: Path, output_file: Path, num_workers: int = None
) -> None:
    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    source_files = [
        path for path in h5_files if path.resolve() != output_file.resolve()
    ]
    if not source_files:
        raise FileNotFoundError(
            f"No source .h5 files found in {input_dir} after excluding output file"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"Processing {len(source_files)} files using {num_workers} workers...")

    # Process files in parallel
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_single_file, source_files),
            total=len(source_files),
            desc="Extracting Data",
        ):
            results.append(result)

    # Write results sequentially to the output HDF5 file
    with h5py.File(output_file, "w") as dst:
        dst.attrs["description"] = (
            "Minimal geothermal wells, field energy summary, and per-well WEPT timeseries"
        )
        dst.attrs["source_dir"] = str(input_dir)

        for res in tqdm(results, desc="Writing HDF5", unit="file"):
            if res["status"] == "skipped":
                tqdm.write(f"Skipped {res['name']}: {res['reason']}")
                continue
            elif res["status"] == "error":
                tqdm.write(f"Error processing {res['name']}: {res['reason']}")
                continue

            sample_group = dst.create_group(res["stem"])
            sample_group.attrs["source_file"] = res["source_file"]

            sample_group.create_dataset(
                "wells", data=res["wells"], compression="gzip", compression_opts=4
            )
            sample_group.create_dataset(
                "well_wept",
                data=res["well_wept"],
                compression="gzip",
                compression_opts=4,
            )

            sample_group.create_dataset(
                "well_vertical_profile",
                data=res["well_vertical_profile"],
                compression="gzip",
                compression_opts=4,
            )
            sample_group.create_dataset(
                "well_tp_profiles",
                data=res["well_tp_profiles"],
                compression="gzip",
                compression_opts=4,
            )

            inputs_group = sample_group.create_group("inputs")
            inputs_group.create_dataset(
                "geology_edge_index",
                data=res["geology_edge_index"],
                compression="gzip",
                compression_opts=4,
            )
            inputs_group.create_dataset(
                "geology_edge_attr",
                data=res["geology_edge_attr"],
                compression="gzip",
                compression_opts=4,
            )

            outputs_group = sample_group.create_group("outputs")
            outputs_group.create_dataset(
                "field_energy_injection_rate",
                data=res["energy_inj"],
                compression="gzip",
                compression_opts=4,
            )
            outputs_group.create_dataset(
                "field_energy_production_rate",
                data=res["energy_prod"],
                compression="gzip",
                compression_opts=4,
            )
            outputs_group.create_dataset(
                "field_energy_production_total",
                data=res["energy_total"],
                compression="gzip",
                compression_opts=4,
            )

            sample_group.attrs["well_count"] = res["well_count"]
            # tqdm.write(f"Processed {res['name']}: wells={res['well_count']}")

    print(f"Wrote minimal compiled dataset to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile minimal geothermal well + field energy data from multiple HDF5 files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing source .h5 files (default: data)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/minimal_compiled.h5"),
        help="Output compiled .h5 file path (default: data/minimal_compiled.h5)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for extraction (default: CPU count - 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compile_dataset(args.input_dir, args.output_file, args.num_workers)


if __name__ == "__main__":
    main()
