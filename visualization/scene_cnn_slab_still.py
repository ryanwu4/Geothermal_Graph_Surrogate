"""CNN Edge Encoder — Slab Region Visualization.

Renders a still image showing the padded 3D bounding box region that the
ContinuousCropper extracts for a selected well pair.  The CNN slab region
is rendered as a semi-transparent shaded box overlaid on the permeability
field.  A* paths between the selected well pair are shown.

The bounding box logic mirrors ``physics_slab.ContinuousCropper``:
  - X: min(x_a, x_b) - 10  …  max(x_a, x_b) + 10
  - Y: min(y_a, y_b) - 10  …  max(y_a, y_b) + 10
  - Z: 0  …  max(z_a, z_b)

Black background.
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
    Line3D,
    Rectangle,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    config,
    interpolate_color,
    ManimColor,
    Prism,
    DashedLine,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
SLICE_RESOLUTON = 80
SLICE_COUNT_PERM = 7
SLICE_OPACITY_PERM = 0.28
WELL_THICKNESS = 0.04
PATH_WIDTH = 0.025
GLOBAL_ZOOM = 1.0

# CNN Cropper padding (grid cells) — must match physics_slab.ContinuousCropper
CROP_PAD_XY = 10

config.disable_caching = True
config.background_color = BLACK

# Palettes
COLOR_LO, COLOR_MID, COLOR_HI = (
    ManimColor("#440154"),
    ManimColor("#21918c"),
    ManimColor("#fde725"),
)

# Slab highlight colour
COLOR_SLAB = ManimColor("#00aaff")
SLAB_OPACITY = 0.15
SLAB_EDGE_COLOR = ManimColor("#00ccff")

# A* path colour
COLOR_ASTAR = ManimColor("#ff0000")
COLOR_ASTAR_OUTSIDE = ManimColor("#ff0000")


def _perm_color(t: float) -> ManimColor:
    t = np.clip(t, 0.0, 1.0)
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2.0)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2.0)


def _make_colorbar(p_lo: float, p_hi: float) -> VGroup:
    bar = Rectangle(height=3, width=0.2, stroke_color=WHITE, stroke_width=1)
    bar.set_fill([COLOR_LO, COLOR_MID, COLOR_HI], opacity=1)
    t_min = Text(f"{p_lo:.1f}", font_size=24, color=WHITE).next_to(bar, DOWN, buff=0.1)
    t_max = Text(f"{p_hi:.1f}", font_size=24, color=WHITE).next_to(bar, UP, buff=0.1)
    t_label = Text("log10(k)", font_size=20, color=WHITE).next_to(t_max, UP, buff=0.2)
    cb = VGroup(bar, t_min, t_max, t_label)
    return cb


def _make_wireframe_box(corners: np.ndarray, color, stroke_width=2.0) -> VGroup:
    """Draw 12 edges of a 3D box given 8 corner points.

    corners: (8, 3) array in Manim space.
    Order: [x_min,y_min,z_min], [x_max,y_min,z_min], [x_max,y_max,z_min],
           [x_min,y_max,z_min], [x_min,y_min,z_max], [x_max,y_min,z_max],
           [x_max,y_max,z_max], [x_min,y_max,z_max]
    """
    edges_idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical edges
    ]
    lines = VGroup()
    for a, b in edges_idx:
        line = Line3D(
            start=corners[a],
            end=corners[b],
            color=color,
            thickness=0.02,
        )
        lines.add(line)
    return lines


def _make_filled_faces(corners: np.ndarray, color, opacity=0.12) -> VGroup:
    """Draw 6 filled faces of a box using Manim Squares positioned in 3D.

    We approximate each face with a flat square at the correct position and orientation.
    """
    faces = VGroup()

    # We'll build each face from 4 corner indices
    face_defs = [
        (0, 1, 2, 3),  # bottom (z_min)
        (4, 5, 6, 7),  # top (z_max)
        (0, 1, 5, 4),  # front (y_min)
        (2, 3, 7, 6),  # back (y_max)
        (0, 3, 7, 4),  # left (x_min)
        (1, 2, 6, 5),  # right (x_max)
    ]

    from manim import Polygon

    for idxs in face_defs:
        pts = [corners[i] for i in idxs]
        face = Polygon(
            *pts,
            fill_color=color,
            fill_opacity=opacity,
            stroke_width=0,
            shade_in_3d=True,
        )
        faces.add(face)

    return faces


class CnnSlabStill(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)

        # -- Permeability slices --------------------------------------------
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
                            fill_opacity=SLICE_OPACITY_PERM,
                            shade_in_3d=True,
                        )
                        sq.move_to(pos)
                        layer.add(sq)
            perm_slices.add(layer)

        # -- Select a well pair for CNN slab visualization ------------------
        well_a, well_b = 11, 6

        # -- Only render the two wells in the selected pair -----------------
        well_lines = VGroup()
        for idx in (well_a, well_b):
            line = Line3D(
                start=data.well_tops[idx],
                end=data.well_bottoms[idx],
                color=BLUE if data.is_injector[idx] else ORANGE,
                thickness=WELL_THICKNESS,
            )
            well_lines.add(line)

        # -- Compute CNN ContinuousCropper bounding box ---------------------
        # The model's pos_xyz uses wells["depth"] = deepest perforated Z layer.
        # data_loader's well_grid_coords uses depth_centroid (shallower), but
        # well_bottom_grid stores the deepest Z — which matches the preprocessed
        # wells table's "depth" field used for pos_xyz in the CNN pipeline.
        coord_a = data.well_bottom_grid[well_a].astype(float)  # (x, y, z_deepest)
        coord_b = data.well_bottom_grid[well_b].astype(float)

        x_min_grid = min(coord_a[0], coord_b[0]) - CROP_PAD_XY
        x_max_grid = max(coord_a[0], coord_b[0]) + CROP_PAD_XY
        y_min_grid = min(coord_a[1], coord_b[1]) - CROP_PAD_XY
        y_max_grid = max(coord_a[1], coord_b[1]) + CROP_PAD_XY
        z_min_grid = 0.0  # surface — ContinuousCropper always starts at 0
        z_max_grid = max(coord_a[2], coord_b[2])

        # Clamp to grid bounds
        x_min_grid = max(x_min_grid, 0)
        x_max_grid = min(x_max_grid, nx - 1)
        y_min_grid = max(y_min_grid, 0)
        y_max_grid = min(y_max_grid, ny - 1)
        z_max_grid = min(z_max_grid, nz - 1)

        # Convert 8 corners from grid to manim space
        box_corners_grid = np.array(
            [
                [x_min_grid, y_min_grid, z_max_grid],  # 0: bottom-front-left (deep)
                [x_max_grid, y_min_grid, z_max_grid],  # 1
                [x_max_grid, y_max_grid, z_max_grid],  # 2
                [x_min_grid, y_max_grid, z_max_grid],  # 3
                [x_min_grid, y_min_grid, z_min_grid],  # 4: top-front-left (surface)
                [x_max_grid, y_min_grid, z_min_grid],  # 5
                [x_max_grid, y_max_grid, z_min_grid],  # 6
                [x_min_grid, y_max_grid, z_min_grid],  # 7
            ]
        )
        box_corners_manim = data.grid_to_manim(box_corners_grid)

        # Build wireframe + translucent faces
        wireframe = _make_wireframe_box(box_corners_manim, SLAB_EDGE_COLOR)
        faces = _make_filled_faces(box_corners_manim, COLOR_SLAB, SLAB_OPACITY)
        slab_box = VGroup(faces, wireframe)

        # -- A* paths between the selected well pair ------------------------
        # Color segments inside the CNN box red, outside yellow.
        selected_pair = {(well_a, well_b), (well_b, well_a)}
        path_mobs = VGroup()

        for path_idx, (path, pair) in enumerate(zip(data.paths, data.path_well_pairs)):
            if tuple(pair) not in selected_pair:
                continue
            p_sub = (
                np.vstack([path[:: max(1, len(path) // 60)], path[-1:]])
                if len(path) > 60
                else path
            )
            polyline = VGroup()
            for j in range(len(p_sub) - 1):
                # Test midpoint in Manim space against box extents
                mid = (p_sub[j] + p_sub[j + 1]) / 2.0
                # Invert grid_to_manim: manim = (grid - offset) * scale; z *= -1
                # So: grid = manim / scale + offset, but unflip Z first
                mid_grid = mid.copy()
                mid_grid[2] *= -1.0  # unflip Z
                mid_grid = mid_grid / data.scale + data.offset
                inside = (
                    x_min_grid <= mid_grid[0] <= x_max_grid
                    and y_min_grid <= mid_grid[1] <= y_max_grid
                    and z_min_grid <= mid_grid[2] <= z_max_grid
                )
                col = COLOR_ASTAR if inside else COLOR_ASTAR_OUTSIDE
                seg = Line3D(
                    start=p_sub[j],
                    end=p_sub[j + 1],
                    color=col,
                    thickness=PATH_WIDTH,
                )
                polyline.add(seg)
            path_mobs.add(polyline)

        # -- Colorbar -------------------------------------------------------
        colorbar = _make_colorbar(p_lo, p_hi)
        colorbar.to_edge(RIGHT, buff=0.5)

        # -- Camera ---------------------------------------------------------
        self.set_camera_orientation(
            phi=75 * np.pi / 180, theta=-45 * np.pi / 180, zoom=GLOBAL_ZOOM
        )

        # -- Add everything -------------------------------------------------
        self.add(perm_slices, well_lines, path_mobs, slab_box)
        self.add_fixed_in_frame_mobjects(colorbar)

        self.wait(0.04)
