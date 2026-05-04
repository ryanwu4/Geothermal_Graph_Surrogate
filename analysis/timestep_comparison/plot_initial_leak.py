#!/usr/bin/env python3
"""Plot Pressure and Temperature: F0000 vs F0001 vs difference.

Consumes the HDF5 dump produced by
``GeologicalSimulationWrapper.jl/scripts/dev_dump_initial_leak.jl`` and
renders a 2-row × 3-column figure for one Z-slice (or a montage of
several Z-slices). Black-background manim-style theme to match the rest
of the analysis pipeline.

Usage:
    python plot_initial_leak.py path/to/leak_check_dump.h5 \
        [--z 32] [--all-z] [--out fig.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Plot styling — manim-ish black background.
# ---------------------------------------------------------------------------
MANIM_BG = "#000000"
MANIM_BLUE = "#58C4DD"
MANIM_ORANGE = "#FF9000"
MANIM_WHITE = "#FFFFFF"
MANIM_GREY = "#888888"

FONT_SIZE = 16
TITLE_SIZE = 15
TICK_SIZE = 13
CBAR_LABEL_SIZE = 13


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "figure.facecolor": MANIM_BG,
            "axes.facecolor": MANIM_BG,
            "axes.edgecolor": MANIM_WHITE,
            "axes.labelcolor": MANIM_WHITE,
            "xtick.color": MANIM_WHITE,
            "ytick.color": MANIM_WHITE,
            "text.color": MANIM_WHITE,
            "savefig.facecolor": MANIM_BG,
        }
    )


def _style_ax(ax) -> None:
    ax.set_facecolor(MANIM_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(MANIM_WHITE)
    ax.tick_params(colors=MANIM_WHITE, labelsize=TICK_SIZE)


def _add_cbar(fig, im, ax, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SIZE, colors=MANIM_WHITE)
    cbar.set_label(label, fontsize=CBAR_LABEL_SIZE, color=MANIM_WHITE)
    cbar.outline.set_edgecolor(MANIM_WHITE)


def _masked(slice2d: np.ndarray, mask2d: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(mask2d <= 0.5, slice2d)


def _plot_one_z(
    P0: np.ndarray, T0: np.ndarray,
    P1: np.ndarray, T1: np.ndarray,
    mask: np.ndarray,
    z: int,
    realization_name: str,
    out_path: Path,
) -> None:
    """Render a single 2x3 figure for depth slice k=z.

    Arrays from HDF5 are (K, J, I) in Python (Julia writes (I, J, K) in Fortran
    order; HDF5 reverses the axes for C-order readers).  Slicing P[z, :, :]
    gives a (J, I) horizontal plane — imshow then shows I along X and J along Y.
    """
    m = mask[z, :, :]
    p0 = _masked(P0[z, :, :], m)
    p1 = _masked(P1[z, :, :], m)
    dp = _masked((P1 - P0)[z, :, :], m)

    t0 = _masked(T0[z, :, :], m)
    t1 = _masked(T1[z, :, :], m)
    dt = _masked((T1 - T0)[z, :, :], m)

    # Shared color limits across F0000 and F0001 for direct comparison.
    p_vmin = float(min(p0.min(), p1.min()))
    p_vmax = float(max(p0.max(), p1.max()))
    t_vmin = float(min(t0.min(), t1.min()))
    t_vmax = float(max(t0.max(), t1.max()))

    # Diff: symmetric around 0 with a divergent cmap.
    dp_lim = float(np.nanmax(np.abs(dp)))
    dt_lim = float(np.nanmax(np.abs(dt)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=MANIM_BG)

    # Row 0: Pressure
    im = axes[0, 0].imshow(p0, origin="lower", cmap="viridis", vmin=p_vmin, vmax=p_vmax)
    axes[0, 0].set_title("Pressure F0000 (true initial)")
    _add_cbar(fig, im, axes[0, 0], "bar")
    im = axes[0, 1].imshow(p1, origin="lower", cmap="viridis", vmin=p_vmin, vmax=p_vmax)
    axes[0, 1].set_title("Pressure F0001 (after 1 report step)")
    _add_cbar(fig, im, axes[0, 1], "bar")
    im = axes[0, 2].imshow(dp, origin="lower", cmap="RdBu_r", vmin=-dp_lim, vmax=dp_lim)
    axes[0, 2].set_title(f"ΔP = F0001 − F0000   (max |ΔP|={dp_lim:.1f})")
    _add_cbar(fig, im, axes[0, 2], "bar")

    # Row 1: Temperature
    im = axes[1, 0].imshow(t0, origin="lower", cmap="inferno", vmin=t_vmin, vmax=t_vmax)
    axes[1, 0].set_title("Temperature F0000 (true initial)")
    _add_cbar(fig, im, axes[1, 0], "°C")
    im = axes[1, 1].imshow(t1, origin="lower", cmap="inferno", vmin=t_vmin, vmax=t_vmax)
    axes[1, 1].set_title("Temperature F0001 (after 1 report step)")
    _add_cbar(fig, im, axes[1, 1], "°C")
    im = axes[1, 2].imshow(dt, origin="lower", cmap="RdBu_r", vmin=-dt_lim, vmax=dt_lim)
    axes[1, 2].set_title(f"ΔT = F0001 − F0000   (max |ΔT|={dt_lim:.2f})")
    _add_cbar(fig, im, axes[1, 2], "°C")

    for ax in axes.ravel():
        _style_ax(ax)
        ax.set_xlabel("I")
        ax.set_ylabel("J")

    fig.suptitle(
        f"{realization_name}  Z={z}",
        fontsize=TITLE_SIZE + 2, color=MANIM_WHITE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=MANIM_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_h5", type=Path, help="HDF5 produced by dev_dump_initial_leak.jl")
    parser.add_argument("--z", type=int, default=None, help="Z-slice index (default: middle of active range)")
    parser.add_argument("--all-z", action="store_true", help="Render one figure per active Z-slice instead of one")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG (default: alongside dump_h5)")
    args = parser.parse_args()

    _set_plot_style()
    with h5py.File(args.dump_h5, "r") as f:
        P0 = f["/F0000/pressure"][...]
        T0 = f["/F0000/temperature"][...]
        P1 = f["/F0001/pressure"][...]
        T1 = f["/F0001/temperature"][...]
        mask = f["/active_mask"][...]
        realization_name = f.attrs.get("realization_name", "unknown")
        if isinstance(realization_name, bytes):
            realization_name = realization_name.decode()

    print(f"loaded grids: P shape={P0.shape}, T shape={T0.shape}, mask shape={mask.shape}")

    # mask shape from HDF5 is (K, J, I); collapse J and I to find active K slices.
    active_z = np.where(mask.any(axis=(1, 2)))[0]
    if active_z.size == 0:
        raise RuntimeError("No active cells in dump — can't pick a K slice.")
    z_lo, z_hi = int(active_z.min()), int(active_z.max())
    print(f"active K-range: [{z_lo}, {z_hi}] of {mask.shape[0]}")

    out_root = args.out if args.out else args.dump_h5.with_suffix("")
    if args.all_z:
        for z in range(z_lo, z_hi + 1):
            _plot_one_z(P0, T0, P1, T1, mask, z, realization_name,
                        out_path=Path(str(out_root) + f"_z{z:03d}.png"))
    else:
        z = args.z if args.z is not None else (z_lo + z_hi) // 2
        out_path = Path(str(out_root) + f"_z{z:03d}.png") if args.z is None else (
            args.out if args.out else Path(str(out_root) + f"_z{z:03d}.png")
        )
        _plot_one_z(P0, T0, P1, T1, mask, z, realization_name, out_path)


if __name__ == "__main__":
    main()
