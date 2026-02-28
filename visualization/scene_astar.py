"""Scene 3 — A* Paths & Transparent Viridis Slices.

Renders A* paths overlayed on high-res seamless Viridis slices.
100% zoom. No headers. Grouped for depth fidelity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    BLUE,
    ORANGE,
    UP,
    WHITE,
    Create,
    FadeIn,
    Line3D,
    ManimColor,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    config,
    interpolate_color,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
SLICE_RESOLUTON = 80
SLICE_COUNT = 8
SLICE_OPACITY = 0.28
WELL_THICKNESS = 0.04
PATH_WIDTH = 0.025
ROTATION_RATE = 0.10
GLOBAL_ZOOM = 1.0

config.disable_caching = True

COLOR_INJ_EXT = ManimColor("#ff0000")
COLOR_INJ_INJ = ManimColor("#ff0000")
COLOR_EXT_EXT = ManimColor("#ff0000")
COLOR_LO, COLOR_MID, COLOR_HI = (
    ManimColor("#440154"),
    ManimColor("#21918c"),
    ManimColor("#fde725"),
)


def _perm_color(t: float) -> ManimColor:
    t = np.clip(t, 0.0, 1.0)
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2)


def _make_transparent_slices(data: ReservoirData) -> VGroup:
    nz, nx, ny = data.grid_shape
    slices = VGroup()
    log_perm_total = np.log10(np.clip(data.perm[data.valid_mask], 1e-18, None))
    p_lo, p_hi = (
        np.percentile(log_perm_total, [2, 98])
        if len(log_perm_total) > 0
        else (-16, -10)
    )
    xi, yi = np.linspace(0, nx - 1, SLICE_RESOLUTON).astype(int), np.linspace(
        0, ny - 1, SLICE_RESOLUTON
    ).astype(int)
    sq_size = max(nx / SLICE_RESOLUTON, ny / SLICE_RESOLUTON) * data.scale
    depths = np.linspace(0, nz - 1, SLICE_COUNT).astype(int)
    for z_layer in depths:
        layer_group = VGroup()
        for i in xi:
            for j in yi:
                if data.valid_mask[z_layer, i, j]:
                    val = data.perm[z_layer, i, j]
                    log_val = np.log10(max(val, 1e-18))
                    t = (log_val - p_lo) / (p_hi - p_lo) if p_hi > p_lo else 0.5
                    pos = data.grid_to_manim(np.array([[i, j, z_layer]]))[0]
                    sq = Square(
                        side_length=sq_size,
                        stroke_width=0,
                        fill_color=_perm_color(float(t)),
                        fill_opacity=SLICE_OPACITY,
                        shade_in_3d=True,
                    )
                    sq.move_to(pos)
                    layer_group.add(sq)
        slices.add(layer_group)
    return slices


def _make_well_lines(data: ReservoirData) -> VGroup:
    wells = VGroup()
    for i in range(len(data.well_tops)):
        top, bot = data.well_tops[i], data.well_bottoms[i]
        line = Line3D(
            start=top,
            end=bot,
            color=BLUE if data.is_injector[i] else ORANGE,
            thickness=WELL_THICKNESS,
        )
        wells.add(line)
    return wells


def _path_polyline(path: np.ndarray, color: ManimColor) -> VGroup:
    lines = VGroup()
    for j in range(len(path) - 1):
        lines.add(
            Line3D(start=path[j], end=path[j + 1], color=color, thickness=PATH_WIDTH)
        )
    return lines


class AStarScene(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)
        self.set_camera_orientation(
            phi=70 * np.pi / 180, theta=-45 * np.pi / 180, zoom=GLOBAL_ZOOM
        )

        bg = _make_transparent_slices(data)
        wells = _make_well_lines(data)

        # Depth fix: group bg/wells
        group = VGroup(bg, wells)
        self.add(group)
        self.play(FadeIn(group, run_time=2))

        self.begin_ambient_camera_rotation(rate=ROTATION_RATE)
        for k, (path, pair) in enumerate(zip(data.paths, data.path_well_pairs)):
            src_idx, dst_idx = pair
            si, di = data.is_injector[src_idx], data.is_injector[dst_idx]
            col = (
                COLOR_INJ_EXT if si != di else (COLOR_INJ_INJ if si else COLOR_EXT_EXT)
            )
            path_sub = (
                np.vstack([path[:: max(1, len(path) // 100)], path[-1:]])
                if len(path) > 100
                else path
            )
            polyline = _path_polyline(path_sub, col)
            self.play(Create(polyline, run_time=0.6))

        self.wait(6)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=30 * np.pi / 180, theta=0, run_time=3)
        self.wait(1)
