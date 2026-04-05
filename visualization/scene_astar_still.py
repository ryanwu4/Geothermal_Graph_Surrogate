"""A* Paths — Still Frame.

Renders a single still image that reproduces the last frame of the
A* Paths segment from MasterScene (before the title fades out).

White background.  One injector and one producer well are labelled.
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
    RIGHT,
    UP,
    WHITE,
    BLACK,
    Circle,
    Create,
    FadeIn,
    Line,
    Line3D,
    Rectangle,
    Scene,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    config,
    interpolate_color,
    ManimColor,
    Arrow3D,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
SLICE_RESOLUTON = 80
SLICE_COUNT_PERM = 7
SLICE_OPACITY_ASTAR = 0.28
WELL_THICKNESS = 0.04
PATH_WIDTH = 0.025
GLOBAL_ZOOM = 1.0

config.disable_caching = True
config.background_color = WHITE

# Palettes
COLOR_LO, COLOR_MID, COLOR_HI = (
    ManimColor("#440154"),
    ManimColor("#21918c"),
    ManimColor("#fde725"),
)

# A* path colour
COLOR_ASTAR = ManimColor("#ff0000")


def _perm_color(t: float) -> ManimColor:
    t = np.clip(t, 0.0, 1.0)
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2.0)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2.0)


def _make_colorbar(p_lo: float, p_hi: float) -> VGroup:
    bar = Rectangle(height=3, width=0.2, stroke_color=BLACK, stroke_width=1)
    bar.set_fill([COLOR_LO, COLOR_MID, COLOR_HI], opacity=1)
    t_min = Text(f"{p_lo:.1f}", font_size=24, color=BLACK).next_to(bar, DOWN, buff=0.1)
    t_max = Text(f"{p_hi:.1f}", font_size=24, color=BLACK).next_to(bar, UP, buff=0.1)
    t_label = Text("log10(k)", font_size=20, color=BLACK).next_to(t_max, UP, buff=0.2)
    cb = VGroup(bar, t_min, t_max, t_label)
    return cb


class AstarStill(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)

        # -- Wells ----------------------------------------------------------
        well_lines = VGroup()
        for i in range(len(data.well_tops)):
            line = Line3D(
                start=data.well_tops[i],
                end=data.well_bottoms[i],
                color=BLUE if data.is_injector[i] else ORANGE,
                thickness=WELL_THICKNESS,
            )
            well_lines.add(line)

        # -- Permeability slices at A* opacity ------------------------------
        nz, nx, ny = data.grid_shape
        log_perm_total = np.log10(np.clip(data.perm[data.valid_mask], 1e-18, None))
        p_lo, p_hi = (
            np.percentile(log_perm_total, [2, 98])
            if len(log_perm_total) > 0
            else (-16, -10)
        )
        depths_p = np.linspace(0, nz - 1, SLICE_COUNT_PERM).astype(int)
        xi = np.linspace(0, nx - 1, SLICE_RESOLUTON).astype(int)
        yi = np.linspace(0, ny - 1, SLICE_RESOLUTON).astype(int)
        sq_size = max(nx / SLICE_RESOLUTON, ny / SLICE_RESOLUTON) * data.scale

        perm_slices = VGroup()
        for zl in depths_p:
            layer = VGroup()
            for i in xi:
                for j in yi:
                    if data.valid_mask[zl, i, j]:
                        val = data.perm[zl, i, j]
                        log_val = np.log10(max(val, 1e-18))
                        t = (log_val - p_lo) / (p_hi - p_lo) if p_hi > p_lo else 0.5
                        pos = data.grid_to_manim(np.array([[i, j, zl]]))[0]
                        sq = Square(
                            side_length=sq_size,
                            stroke_width=0,
                            fill_color=_perm_color(float(t)),
                            fill_opacity=SLICE_OPACITY_ASTAR,
                            shade_in_3d=True,
                        )
                        sq.move_to(pos)
                        layer.add(sq)
            perm_slices.add(layer)

        # -- A* paths -------------------------------------------------------
        path_mobs = VGroup()
        for k, (path, pair) in enumerate(zip(data.paths, data.path_well_pairs)):
            col = COLOR_ASTAR
            p_sub = (
                np.vstack([path[:: max(1, len(path) // 60)], path[-1:]])
                if len(path) > 60
                else path
            )
            polyline = VGroup()
            for j in range(len(p_sub) - 1):
                seg = Line3D(
                    start=p_sub[j], end=p_sub[j + 1], color=col, thickness=PATH_WIDTH
                )
                polyline.add(seg)
            path_mobs.add(polyline)



        # -- Colorbar -------------------------------------------------------
        colorbar = _make_colorbar(p_lo, p_hi)
        colorbar.to_edge(RIGHT, buff=0.5)

        # -- Well labels: pick the most separated injector & producer -------
        inj_indices = [i for i in range(len(data.is_injector)) if data.is_injector[i]]
        prod_indices = [i for i in range(len(data.is_injector)) if not data.is_injector[i]]

        # Pick the pair with maximum distance so labels don't overlap
        best_dist, inj_idx, prod_idx = -1, None, None
        for ii in inj_indices:
            for pi in prod_indices:
                d = np.linalg.norm(data.well_tops[ii] - data.well_tops[pi])
                if d > best_dist:
                    best_dist, inj_idx, prod_idx = d, ii, pi

        well_label_mobs = []
        if inj_idx is not None:
            lbl_inj = Text("Injector", font_size=36, color=BLUE, weight="BOLD")
            lbl_pos = data.well_tops[inj_idx].copy()
            lbl_pos[2] += 0.55
            lbl_inj.move_to(lbl_pos)
            well_label_mobs.append(lbl_inj)

        if prod_idx is not None:
            lbl_prod = Text("Producer", font_size=36, color=ORANGE, weight="BOLD")
            lbl_pos = data.well_tops[prod_idx].copy()
            lbl_pos[2] += 0.55
            lbl_prod.move_to(lbl_pos)
            well_label_mobs.append(lbl_prod)

        # -- Camera ---------------------------------------------------------
        self.set_camera_orientation(
            phi=75 * np.pi / 180, theta=-45 * np.pi / 180, zoom=GLOBAL_ZOOM
        )

        # -- Add everything as a still frame --------------------------------
        self.add(perm_slices, well_lines, path_mobs)
        self.add_fixed_in_frame_mobjects(colorbar)

        # Labels pinned in 3D space but always face the camera
        for lbl in well_label_mobs:
            self.add_fixed_orientation_mobjects(lbl)

        # Hold for one frame so Manim writes the image
        self.wait(0.04)
