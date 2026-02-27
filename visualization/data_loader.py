"""Load 3D reservoir grids, wells, and A* paths from a raw geothermal HDF5 file.

All coordinates are returned centred at the origin and scaled to fit
Manim's coordinate system (roughly ±3 units per axis).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Allow importing geology_graph from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from geothermal.geology_graph import generate_geology_edges  # noqa: E402


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class ReservoirData:
    """Container for all visualisation-relevant data extracted from one HDF5 case."""

    # 3-D grids clipped to top ``max_z`` layers.  Shape: (Z, X, Y)
    perm: np.ndarray
    porosity: np.ndarray
    temp: np.ndarray
    press: np.ndarray

    # Boolean mask – True where cells are *valid* (non-fault / non-sealed)
    valid_mask: np.ndarray

    # Well info ---
    # (N_wells, 3) – Manim-space (x_m, y_m, z_m) positions
    well_positions: np.ndarray
    # (N_wells,) – True if injector
    is_injector: np.ndarray
    # (N_wells, 3) – original grid indices (x_grid, y_grid, z_grid)
    well_grid_coords: np.ndarray
    # (N_wells, 3) – Manim-space vertical extents (x_m, y_m, z_top_m) and (x_m, y_m, z_bot_m)
    well_tops: np.ndarray
    well_bottoms: np.ndarray
    # (N_wells, 3) – original grid indices (x, y, z) for tops and bottoms
    well_top_grid: np.ndarray
    well_bottom_grid: np.ndarray

    # A* paths – list of (L_i, 3) arrays in Manim space
    paths: list[np.ndarray] = field(default_factory=list)
    # Corresponding edge list  (start_well_idx, goal_well_idx)
    path_well_pairs: list[tuple[int, int]] = field(default_factory=list)

    # Edge attributes from A* (num_edges, 14)
    edge_attr: np.ndarray = field(default_factory=lambda: np.empty((0, 14)))

    # Coordinate-transform helpers
    grid_shape: tuple[int, int, int] = (0, 0, 0)
    scale: float = 1.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def grid_to_manim(self, coords: np.ndarray) -> np.ndarray:
        """Convert grid (x, y, z) indices → Manim (x_m, y_m, z_m) positions.

        Note: Z is flipped so that grid z=0 (top) is positive in Manim.
        """
        # coords is (..., 3) -> (x, y, z)
        m_coords = (coords.astype(np.float64) - self.offset) * self.scale
        # Flip Z: z_m_flipped = -z_m
        m_coords[..., 2] *= -1.0
        return m_coords


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

DEFAULT_MAX_Z = 70
MANIM_HALF_WIDTH = 3.0  # target: scene fits in roughly ±3 units


def load(
    h5_path: str | Path,
    max_z: int | None = None,
    k_neighbors: int = 2,
) -> ReservoirData:
    """Read a raw geothermal HDF5 file and return *ReservoirData*.

    Parameters
    ----------
    h5_path : path to a raw ``v2.5_*.h5`` file (not the compiled minimal one).
    max_z : if provided, clips to this depth. Otherwise, clips to deepest well + 5.
    k_neighbors : neighbour count passed to ``generate_geology_edges``.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as src:
        # 1. Determine Clipping Depth
        if max_z is None:
            # Read only IsWell to find the deepest point
            is_well_full = src["Input/IsWell"][:]
            well_indices = np.where(is_well_full == 1)
            if len(well_indices[0]) > 0:
                max_well_z = int(well_indices[0].max())
                max_z = max_well_z + 5
            else:
                max_z = DEFAULT_MAX_Z

        # 2. Read clipped grids (Z, X, Y)
        perm_x = src["Input/PermX"][:max_z].astype(np.float32)
        perm_y = src["Input/PermY"][:max_z].astype(np.float32)
        perm_z = src["Input/PermZ"][:max_z].astype(np.float32)
        porosity = src["Input/Porosity"][:max_z].astype(np.float32)
        temp0 = src["Input/Temperature0"][:max_z].astype(np.float32)
        press0 = src["Input/Pressure0"][:max_z].astype(np.float32)
        is_well = src["Input/IsWell"][:max_z]
        inj_rate = src["Input/InjRate"][:max_z]

    perm_avg = (perm_x + perm_y + perm_z) / 3.0
    valid_mask = (perm_avg > 1e-15) & (porosity > 0.0)

    nz, nx, ny = perm_avg.shape

    # --- Wells --------------------------------------------------------------
    well_mask = is_well == 1
    has_well_xy = np.any(well_mask, axis=0)
    w_x, w_y = np.where(has_well_xy)
    n_wells = len(w_x)

    # Min/Max Z for each well
    w_min_z = np.zeros(n_wells, dtype=np.int32)
    w_max_z = np.zeros(n_wells, dtype=np.int32)
    depth_centroid = np.zeros(n_wells, dtype=np.int32)
    is_injector = np.zeros(n_wells, dtype=bool)

    for i in range(n_wells):
        xi, yi = w_x[i], w_y[i]
        z_perf = np.where(well_mask[:, xi, yi])[0]
        if len(z_perf) == 0:
            w_min_z[i] = w_max_z[i] = depth_centroid[i] = 0
            is_injector[i] = False
            continue

        w_min_z[i] = z_perf.min()
        w_max_z[i] = z_perf.max()

        # Injection rate at deepest layer to determine type
        w_inj_rate = inj_rate[w_max_z[i], xi, yi]
        is_injector[i] = w_inj_rate > 0

        # Perm-weighted centroid (same logic as compile_minimal_geothermal_h5.py)
        if len(z_perf) <= 1:
            depth_centroid[i] = w_max_z[i]
        else:
            perms = perm_avg[z_perf, xi, yi].astype(np.float64)
            perms = np.maximum(perms, 1e-30)
            centroid_z = np.average(z_perf.astype(np.float64), weights=perms)
            depth_centroid[i] = z_perf[np.argmin(np.abs(z_perf - centroid_z))]

    well_grid_coords = np.stack([w_x, w_y, depth_centroid], axis=1)
    well_top_grid = np.stack([w_x, w_y, w_min_z], axis=1)
    well_bottom_grid = np.stack([w_x, w_y, w_max_z], axis=1)

    # --- A* paths -----------------------------------------------------------
    edge_index, edge_attr, path_coords_list = generate_geology_edges(
        perm_avg,
        porosity,
        temp0,
        press0,
        is_injector,
        well_grid_coords,
        k_neighbors=k_neighbors,
        return_paths=True,
    )

    # Build (start, goal) pair list aligned with path_coords_list
    path_well_pairs = []
    if edge_index.shape[1] > 0:
        for col in range(edge_index.shape[1]):
            path_well_pairs.append((int(edge_index[0, col]), int(edge_index[1, col])))

    # --- Coordinate transform -----------------------------------------------
    # Map grid indices to Manim coordinates: centre at origin, fit in ±3 units.
    # Grid coords are (x, y, z) where x ∈ [0, nx), y ∈ [0, ny), z ∈ [0, nz).
    max_dim = max(nx, ny, nz)
    scale = 2.0 * MANIM_HALF_WIDTH / max_dim
    offset = np.array([nx / 2.0, ny / 2.0, nz / 2.0], dtype=np.float64)

    def _to_manim(coords: np.ndarray) -> np.ndarray:
        m_coords = (coords.astype(np.float64) - offset) * scale
        # Flip Z so top (z=0) is positive
        m_coords[..., 2] *= -1.0
        return m_coords

    well_positions = _to_manim(well_grid_coords)
    well_tops = _to_manim(well_top_grid)
    well_bottoms = _to_manim(well_bottom_grid)

    paths_manim: list[np.ndarray] = []
    for p in path_coords_list:
        # p is (L, 3) in (x, y, z) grid coords
        paths_manim.append(_to_manim(p))

    return ReservoirData(
        perm=perm_avg,
        porosity=porosity,
        temp=temp0,
        press=press0,
        valid_mask=valid_mask,
        well_positions=well_positions,
        well_tops=well_tops,
        well_bottoms=well_bottoms,
        well_top_grid=well_top_grid,
        well_bottom_grid=well_bottom_grid,
        is_injector=is_injector,
        well_grid_coords=well_grid_coords,
        paths=paths_manim,
        path_well_pairs=path_well_pairs,
        edge_attr=edge_attr,
        grid_shape=(nz, nx, ny),
        scale=scale,
        offset=offset,
    )
