"""Scene 2 — Permeability Slices (High-Res) & Well Lines.

Renders horizontal slice planes with 80x80 resolution and Viridis colormap.
100% zoom. No headers. Transparent slices respect well positions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    BLUE,
    DOWN,
    LEFT,
    ORANGE,
    ORIGIN,
    RIGHT,
    UP,
    WHITE,
    FadeIn,
    Line3D,
    Rectangle,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    config,
    interpolate_color,
    ManimColor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
# v2.5.1: Significantly higher resolution
SLICE_RESOLUTON = 80
SLICE_COUNT = 7
SLICE_OPACITY = 0.72
WELL_THICKNESS = 0.04
ROTATION_RATE = 0.12
# v2.5.1: Full 100% zoom
GLOBAL_ZOOM = 1.0

config.disable_caching = True

# Viridis Colormap
COLOR_LO = ManimColor("#440154")  # Purple
COLOR_MID = ManimColor("#21918c")  # Teal
COLOR_HI = ManimColor("#fde725")  # Yellow


def _perm_color(t: float) -> ManimColor:
    t = np.clip(t, 0.0, 1.0)
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2.0)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2.0)


def _make_filtered_slices(data: ReservoirData) -> VGroup:
    """Create horizontal slice planes with percentile scaling and seamless high-res tiling."""
    nz, nx, ny = data.grid_shape
    slices = VGroup()

    log_perm_total = np.log10(np.clip(data.perm[data.valid_mask], 1e-18, None))
    if len(log_perm_total) > 0:
        p_lo, p_hi = np.percentile(log_perm_total, [2, 98])
    else:
        p_lo, p_hi = -16.0, -10.0

    xi = np.linspace(0, nx - 1, SLICE_RESOLUTON).astype(int)
    yi = np.linspace(0, ny - 1, SLICE_RESOLUTON).astype(int)

    # Seamless tiling with 5% overlap to prevent gaps
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
        top = data.well_tops[i]
        bot = data.well_bottoms[i]
        colour = BLUE if data.is_injector[i] else ORANGE
        line = Line3D(start=top, end=bot, color=colour, thickness=WELL_THICKNESS)
        wells.add(line)
    return wells


def _make_colorbar(p_lo: float, p_hi: float) -> VGroup:
    bar = Rectangle(height=3, width=0.2, stroke_color=WHITE, stroke_width=1)
    bar.set_fill([COLOR_LO, COLOR_MID, COLOR_HI], opacity=1)
    t_min = Text(f"{p_lo:.1f}", font_size=14).next_to(bar, DOWN, buff=0.1)
    t_max = Text(f"{p_hi:.1f}", font_size=14).next_to(bar, UP, buff=0.1)
    cb = VGroup(bar, t_min, t_max)
    return cb


class PermeabilityScene(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)

        self.set_camera_orientation(
            phi=70 * np.pi / 180, theta=-50 * np.pi / 180, zoom=GLOBAL_ZOOM
        )

        slices = _make_filtered_slices(data)
        wells = _make_well_lines(data)

        log_perm_val = np.log10(np.clip(data.perm[data.valid_mask], 1e-18, None))
        p_lo, p_hi = (
            np.percentile(log_perm_val, [2, 98])
            if len(log_perm_val) > 0
            else (-16, -10)
        )

        colorbar = _make_colorbar(p_lo, p_hi)
        self.add_fixed_in_frame_mobjects(colorbar)
        colorbar.to_edge(RIGHT, buff=0.5)

        # v2.5.1: To fix "spatial inside" transparency, we add wells AFTER the slices.
        # Manim CE depth sorting usually works better when opaque/occluding objects are added last
        # or grouped. We'll group them for consistent 3D handling.
        group = VGroup(slices, wells)
        self.add(group)
        self.play(FadeIn(group, run_time=2.5))

        self.begin_ambient_camera_rotation(rate=ROTATION_RATE)
        self.wait(12)
        self.stop_ambient_camera_rotation()

        self.move_camera(phi=20 * np.pi / 180, theta=0, run_time=3)
        self.wait(1)
