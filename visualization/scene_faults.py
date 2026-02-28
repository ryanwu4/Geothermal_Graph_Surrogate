"""Scene 1 — Fault Structure (High-Res) & Well Lines.

Renders orange fault slices with 80x80 resolution.
100% zoom. No headers. Respects spatial transparency.
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
    FadeIn,
    Line3D,
    Square,
    ThreeDScene,
    VGroup,
    config,
    ManimColor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
# v2.5.1: High fidelity
SLICE_RESOLUTON = 80
FAULT_COLOR = ManimColor("#ff5722")
FAULT_OPACITY = 0.5
WELL_THICKNESS = 0.04
ROTATION_RATE = 0.12
GLOBAL_ZOOM = 1.0

config.disable_caching = True


def _make_fault_slices(data: ReservoirData) -> VGroup:
    """Create high-res horizontal slices where invalid cells are highlighted orange."""
    nz, nx, ny = data.grid_shape
    start_z = int(data.well_top_grid[:, 2].max()) + 20
    end_z = nz - 1
    if start_z >= end_z:
        start_z = nz // 2
    depths = np.linspace(start_z, end_z, 10).astype(int)

    slices = VGroup()
    xi = np.linspace(0, nx - 1, SLICE_RESOLUTON).astype(int)
    yi = np.linspace(0, ny - 1, SLICE_RESOLUTON).astype(int)
    sq_size = max(nx / SLICE_RESOLUTON, ny / SLICE_RESOLUTON) * data.scale

    for z_layer in depths:
        layer_group = VGroup()
        for i in xi:
            for j in yi:
                if not data.valid_mask[z_layer, i, j]:
                    pos = data.grid_to_manim(np.array([[i, j, z_layer]]))[0]
                    sq = Square(
                        side_length=sq_size,
                        stroke_width=0,
                        fill_color=FAULT_COLOR,
                        fill_opacity=FAULT_OPACITY,
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
        line = Line3D(
            start=top,
            end=bot,
            color=BLUE if data.is_injector[i] else ORANGE,
            thickness=WELL_THICKNESS,
        )
        wells.add(line)
    return wells


class FaultScene(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)
        self.set_camera_orientation(
            phi=75 * np.pi / 180, theta=-45 * np.pi / 180, zoom=GLOBAL_ZOOM
        )

        fault_slices = _make_fault_slices(data)
        wells = _make_well_lines(data)

        # Depth sorting fix
        group = VGroup(fault_slices, wells)
        self.add(group)
        self.play(FadeIn(group, run_time=2))

        self.begin_ambient_camera_rotation(rate=ROTATION_RATE)
        self.wait(12)
        self.stop_ambient_camera_rotation()

        self.move_camera(phi=15 * np.pi / 180, theta=0, run_time=3)
        self.wait(1)
