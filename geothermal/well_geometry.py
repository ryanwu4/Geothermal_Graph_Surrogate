"""Deviated-well geometry and proxy-NPV assembly (pure numpy).

Single source of truth for the proxy-NPV objective used by both the active-learning
acquisition loop (surrogate CMA) and the direct-CMA-over-INTERSECT baseline. Gradient-free
(numpy); the differentiable Adam/inference path is intentionally not covered here.

Well-length model (3-segment "deviated / angled subsurface line"), per well:
  (1) 1000 m vertical lead straight down from the nearest surface facility (TVD 0 -> vertical_lead_m)
  (2) straight 3D diagonal from the bottom of that lead to the reservoir TOP at the well (x,y)
  (3) vertical through the reservoir from the reservoir top to the well bottom (k_idx)
  well_length = vertical_lead_m + ||V - R|| + max(0, TVD_bottom - TVD_reservoir_top)
where V = (fac_x, fac_y, vertical_lead_m) and R = (well_x, well_y, TVD_reservoir_top).

Physical coordinates come from the structural corner-point grid extracted from smallerModel.jld2.
The cube is read by h5py with REVERSED axes relative to Julia, so its arrays are (k, j, i) =
(NK, NJ, NI) and are indexed [k-1, j-1, i-1]. CZ is absolute true vertical depth (positive-down,
CZ=0 ~ ground surface). The CMA optimization axes map as i = round(y)+1, j = round(x)+1,
k = round(z)+1 -- identical to the snapshot writer in orchestrator/acquire.py, so the geometry
used to score a candidate matches the well configuration that is emitted to INTERSECT.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .economics import compute_real_discount_rate

DEFAULT_FACILITIES_IJ: list[tuple[int, int]] = [(20, 30), (40, 40)]
DEFAULT_VERTICAL_LEAD_M = 1000.0
DEFAULT_KSURF = 2          # surface datum layer (1-based); matches Julia `top=2`
DEFAULT_PORO_THRESH = 0.01
DEFAULT_RESERVOIR_TOP_K = 11   # fallback when a column has no porous cell

# Required economics keys for npv assembly (revenue comes from the surrogate / IX, not here).
_REQUIRED_NPV_KEYS = (
    "PLANNING_YEARS",
    "WELL_COST_PER_DISTANCE",
    "CAPEX_FLOWLINES_BETWEEN_LOCATIONS_PER_DISTANCE",
    "OPEX_WATER_INJECTOR",
    "OPEX_WATER_PRODUCER",
    "OPEX_ACTIVE_INJECTOR_PER_M3_WATER",
    "OPEX_ACTIVE_PRODUCER_PER_M3_WATER",
    "INJECTOR_RATE_CONSTANT",
    "PRODUCER_RATE_CONSTANT",
    "ANNUAL_WATER_RATE_SCALE",
)


# ---------------------------------------------------------------------------- cube + maps
def load_geo_coord_cube(h5_path: str | Path) -> dict[str, np.ndarray]:
    """Load the structural coordinate cube. Returns CX/CY/CZ each shaped (NK, NJ, NI)=(k,j,i).

    CZ is absolute TVD in metres (positive-down). Geology-independent (one structural grid).
    """
    with h5py.File(str(h5_path), "r") as f:
        cube = {"CX": f["CX"][:], "CY": f["CY"][:], "CZ": f["CZ"][:]}
    if not (cube["CX"].shape == cube["CY"].shape == cube["CZ"].shape):
        raise ValueError("geo coord cube CX/CY/CZ must share a shape")
    if cube["CX"].ndim != 3:
        raise ValueError(f"geo coord cube must be 3-D (k,j,i); got {cube['CX'].shape}")
    return cube


def reservoir_top_k_map(
    porosity_kji: np.ndarray,
    poro_thresh: float = DEFAULT_PORO_THRESH,
    k_default: int = DEFAULT_RESERVOIR_TOP_K,
) -> np.ndarray:
    """Per-column shallowest reservoir layer.

    porosity_kji has shape (NK, NJ, NI). Returns int array shaped (NJ, NI), 1-based k of the first
    layer with porosity > poro_thresh (mirrors Julia find_top_reservoir). Columns with no porous
    cell fall back to ``k_default``.
    """
    if porosity_kji.ndim != 3:
        raise ValueError(f"porosity must be (k,j,i); got {porosity_kji.shape}")
    above = porosity_kji > poro_thresh           # (k, j, i)
    any_above = above.any(axis=0)                # (j, i)
    topk = np.argmax(above, axis=0) + 1          # (j, i), 1-based; argmax->0 when none
    topk = np.where(any_above, topk, k_default)
    return topk.astype(np.int32)                 # index as [j-1, i-1]


def facilities_surface_xy(
    cube: dict[str, np.ndarray],
    facilities_ij: Sequence[Sequence[int]],
    ksurf: int = DEFAULT_KSURF,
) -> np.ndarray:
    """Physical (x, y) of each facility column at the surface datum. Returns (F, 2)."""
    CX, CY = cube["CX"], cube["CY"]
    out = np.empty((len(facilities_ij), 2), dtype=np.float64)
    for f, (i, j) in enumerate(facilities_ij):
        out[f, 0] = CX[ksurf - 1, int(j) - 1, int(i) - 1]
        out[f, 1] = CY[ksurf - 1, int(j) - 1, int(i) - 1]
    return out


def surface_flowline_length(fac_surf_xy: np.ndarray) -> float:
    """Total horizontal surface flowline connecting the fixed facilities.

    Sum of pairwise horizontal distances / 2 (mirrors Julia `sum(fac_distances)/2`). For the two
    fixed facilities this equals the single horizontal distance between them. Constant per run.
    """
    F = fac_surf_xy.shape[0]
    total = 0.0
    for a in range(F):
        for b in range(F):
            if a != b:
                total += float(np.hypot(fac_surf_xy[a, 0] - fac_surf_xy[b, 0],
                                        fac_surf_xy[a, 1] - fac_surf_xy[b, 1]))
    return total / 2.0


# ---------------------------------------------------------------------------- well length
def compute_angled_well_length(
    coords_xyz: np.ndarray,
    *,
    cube: dict[str, np.ndarray],
    fac_surf_xy: np.ndarray,
    reservoir_top_k_map: np.ndarray,
    vertical_lead_m: float = DEFAULT_VERTICAL_LEAD_M,
    ksurf: int = DEFAULT_KSURF,
) -> np.ndarray:
    """3-segment deviated well length per well, in metres.

    coords_xyz: (num_wells, 3) optimization coords (x, y, z). Mapping to 1-based grid:
    i = round(y)+1, j = round(x)+1, k = round(z)+1 (indices clamped to the grid).
    Returns (num_wells,) float64.
    """
    CX, CY, CZ = cube["CX"], cube["CY"], cube["CZ"]
    nk, nj, ni = CZ.shape
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    n = coords_xyz.shape[0]
    out = np.empty(n, dtype=np.float64)
    for w in range(n):
        x, y, z = coords_xyz[w]
        i = min(max(int(round(float(y))) + 1, 1), ni)
        j = min(max(int(round(float(x))) + 1, 1), nj)
        k = min(max(int(round(float(z))) + 1, 1), nk)
        ktop = int(reservoir_top_k_map[j - 1, i - 1])
        ktop = min(max(ktop, 1), nk)
        wx = CX[ksurf - 1, j - 1, i - 1]
        wy = CY[ksurf - 1, j - 1, i - 1]
        tvd_top = CZ[ktop - 1, j - 1, i - 1]
        tvd_bot = CZ[k - 1, j - 1, i - 1]
        dh = np.hypot(fac_surf_xy[:, 0] - wx, fac_surf_xy[:, 1] - wy)
        fi = int(np.argmin(dh))
        fx, fy = fac_surf_xy[fi]
        diag = float(np.sqrt((fx - wx) ** 2 + (fy - wy) ** 2 + (vertical_lead_m - tvd_top) ** 2))
        res_vert = max(0.0, float(tvd_bot - tvd_top))
        out[w] = float(vertical_lead_m) + diag + res_vert
    return out


# ---------------------------------------------------------------------------- npv assembly
def load_npv_terms(economics: dict | str | Path) -> dict[str, Any]:
    """Resolve npv terms from an economics dict (or a path to economics.json).

    Adds REAL_DISCOUNT_RATE (via geothermal.economics.compute_real_discount_rate) and validates
    that all required keys are present. CAPEX_SURFACE_FACILITIES / OPEX_RATE_FROM_CAPEX_SURFACE_FACILITIES
    default to 0.0 (i.e. excluded) when absent. All numbers are read from config -- nothing hardcoded.
    """
    if isinstance(economics, (str, Path)):
        with open(economics, "r") as f:
            economics = json.load(f)
    terms = dict(economics)
    terms["REAL_DISCOUNT_RATE"] = compute_real_discount_rate(terms)
    missing = [k for k in _REQUIRED_NPV_KEYS if k not in terms]
    if missing:
        raise ValueError(f"economics config missing required NPV terms: {missing}")
    terms.setdefault("CAPEX_SURFACE_FACILITIES", 0.0)
    terms.setdefault("OPEX_RATE_FROM_CAPEX_SURFACE_FACILITIES", 0.0)
    return terms


def compute_npv(
    discounted_revenue: float,
    well_lengths: np.ndarray,
    is_injector: Sequence[bool] | np.ndarray,
    *,
    flowline_between_m: float,
    npv_terms: dict[str, Any],
) -> dict[str, float]:
    """Assemble proxy NPV from (discounted) revenue and the cost terms.

    NPV = revenue
          - sum(well_lengths) * WELL_COST_PER_DISTANCE                       (CAPEX wells, deviated)
          - flowline_between_m * CAPEX_FLOWLINES_BETWEEN_LOCATIONS_PER_DISTANCE  (surface flowline, const)
          - CAPEX_SURFACE_FACILITIES                                          (const)
          - discounted_OPEX
    discounted_OPEX = (fixed_water + active_water_proxy + surface_facilities_opex) * sum_{t=1..N}(1/(1+r))^t

    CAPEX is year-0 (undiscounted); OPEX discounted over years 1..N at the real rate. The active-water
    term is a constant-rate proxy (the surrogate has no per-year water rates). Returns a breakdown dict.
    """
    t = npv_terms
    is_inj = np.asarray(is_injector, dtype=bool)
    n_inj = int(is_inj.sum())
    n_prod = int((~is_inj).sum())

    capex_wells = float(np.sum(well_lengths)) * float(t["WELL_COST_PER_DISTANCE"])
    capex_flowline = float(flowline_between_m) * float(t["CAPEX_FLOWLINES_BETWEEN_LOCATIONS_PER_DISTANCE"])
    capex_surface = float(t.get("CAPEX_SURFACE_FACILITIES", 0.0))

    fixed_opex = n_inj * float(t["OPEX_WATER_INJECTOR"]) + n_prod * float(t["OPEX_WATER_PRODUCER"])
    active_opex = (
        n_inj * float(t["INJECTOR_RATE_CONSTANT"]) * float(t["ANNUAL_WATER_RATE_SCALE"])
        * float(t["OPEX_ACTIVE_INJECTOR_PER_M3_WATER"])
        + n_prod * abs(float(t["PRODUCER_RATE_CONSTANT"])) * float(t["ANNUAL_WATER_RATE_SCALE"])
        * float(t["OPEX_ACTIVE_PRODUCER_PER_M3_WATER"])
    )
    surface_opex = float(t.get("OPEX_RATE_FROM_CAPEX_SURFACE_FACILITIES", 0.0)) * capex_surface
    annual_opex = fixed_opex + active_opex + surface_opex

    r = float(t["REAL_DISCOUNT_RATE"])
    n_years = int(t["PLANNING_YEARS"])
    years = np.arange(1, n_years + 1, dtype=np.float64)
    discount_sum = float(np.sum((1.0 / (1.0 + r)) ** years))
    discounted_opex = annual_opex * discount_sum

    total_capex = capex_wells + capex_flowline + capex_surface
    npv = float(discounted_revenue) - total_capex - discounted_opex
    return {
        "npv": npv,
        "revenue": float(discounted_revenue),
        "capex_wells": capex_wells,
        "capex_flowline": capex_flowline,
        "capex_surface": capex_surface,
        "total_capex": total_capex,
        "annual_opex": annual_opex,
        "discounted_opex": discounted_opex,
    }
