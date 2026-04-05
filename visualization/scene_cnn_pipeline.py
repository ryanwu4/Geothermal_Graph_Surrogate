"""CNN Pipeline Visualization — clean rewrite.

Renders a still image showing the edge-feature extraction pipeline
from left to right:

    Physics Slab  →  CNN Prisms  →  FNN  →  Edge Embedding

Layout strategy
---------------
* Manim's default frame is 14.2 wide × 8 tall (config.frame_width/height).
* We divide the horizontal space into 4 *slots* with equal centres.
* Each component is built, then placed at its slot centre.
* Labels sit directly below their component, at a fixed Y.
* Arrows run between adjacent slots' right/left edges with a small gap.
* The 3-D reservoir is rendered by a 3-D camera but all overlay items
  (arrows, labels, graph) are added as fixed-in-frame mobjects so they
  stay screen-aligned regardless of the camera orientation.
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
    OUT,
    WHITE,
    BLACK,
    Line3D,
    Square,
    Text,
    ThreeDScene,
    VGroup,
    config,
    interpolate_color,
    ManimColor,
    Arrow,
    Circle,
    Line,
    Prism,
    DEGREES,
    ORIGIN,
    RoundedRectangle,
    Rectangle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
SLICE_RESOLUTON = 80
SLICE_COUNT_PERM = 7
SLICE_OPACITY_PERM = 0.28
WELL_THICKNESS = 0.04
PATH_WIDTH = 0.025
GLOBAL_ZOOM = 1.0

CROP_PAD_XY = 10

config.disable_caching = True
config.background_color = BLACK
config.save_last_frame = True
config.format = "png"

# Palettes
COLOR_LO, COLOR_MID, COLOR_HI = (
    ManimColor("#440154"),
    ManimColor("#21918c"),
    ManimColor("#fde725"),
)
COLOR_SLAB = ManimColor("#00aaff")
SLAB_OPACITY = 0.15
SLAB_EDGE_COLOR = ManimColor("#00ccff")
COLOR_ASTAR = ManimColor("#ff0000")
COLOR_ASTAR_OUTSIDE = ManimColor("#ffcc00")
COLOR_INJ_EXT = ManimColor("#00e5ff")

# ---------------------------------------------------------------------------
# Layout constants — designed so nothing goes out of frame.
# Frame: x ∈ [-7.1, 7.1], y ∈ [-4, 4].
# ---------------------------------------------------------------------------
# 4 slots dynamically computed in construct().
# Y-centre for components (shifted up a bit to leave room for labels below)
COMP_Y = 0.5
# Y for labels (below the components)
LABEL_Y = -2.8
# Arrow Y (same as component centre)
ARROW_Y = COMP_Y


def _perm_color(t: float) -> ManimColor:
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2.0)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2.0)


# ---------------------------------------------------------------------------
# 3-D helpers (wireframe box + translucent faces for the slab highlight)
# ---------------------------------------------------------------------------


def _make_wireframe_box(corners: np.ndarray, color, stroke_width=2.0) -> VGroup:
    edges_idx = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # vertical
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
    from manim import Polygon

    face_defs = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
    ]
    faces = VGroup()
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


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class CnnPipeline(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)

        # Use predefined slots for 2D visual alignments.
        # Physics Slab, CNN, FNN, Graph
        SCREEN_XS = [-4.9, -1.7, 1.4, 4.4]

        def get_3d_pos(screen_x):
            # To avoid perspective distortion and angle changes (where moving far left causes
            # objects to pitch up/down), we move them exactly along the camera's local X-axis.
            # In a phi=75, theta=-45 camera, moving along the line Y_world = X_world guarantees
            # Y_screen = 0 and perfectly horizontal lateral panning!
            val = screen_x / 1.41421356
            return np.array([val, val, 0.0])

        # ==================================================================
        # 1. PHYSICS SLAB (3-D, slot 0)
        # ==================================================================
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
                            fill_color=_perm_color(t),
                            fill_opacity=SLICE_OPACITY_PERM,
                            shade_in_3d=True,
                        ).move_to(pos)
                        layer.add(sq)
            perm_slices.add(layer)

        # Wells
        well_a, well_b = 11, 6
        well_lines = VGroup()
        for idx in (well_a, well_b):
            line = Line3D(
                start=data.well_tops[idx],
                end=data.well_bottoms[idx],
                color=BLUE if data.is_injector[idx] else ORANGE,
                thickness=WELL_THICKNESS,
            )
            well_lines.add(line)

        # Slab bounding box
        coord_a = data.well_bottom_grid[well_a].astype(float)
        coord_b = data.well_bottom_grid[well_b].astype(float)
        x_min_g = max(min(coord_a[0], coord_b[0]) - CROP_PAD_XY, 0)
        x_max_g = min(max(coord_a[0], coord_b[0]) + CROP_PAD_XY, nx - 1)
        y_min_g = max(min(coord_a[1], coord_b[1]) - CROP_PAD_XY, 0)
        y_max_g = min(max(coord_a[1], coord_b[1]) + CROP_PAD_XY, ny - 1)
        z_min_g = 0.0
        z_max_g = min(max(coord_a[2], coord_b[2]), nz - 1)

        box_corners_grid = np.array(
            [
                [x_min_g, y_min_g, z_max_g],
                [x_max_g, y_min_g, z_max_g],
                [x_max_g, y_max_g, z_max_g],
                [x_min_g, y_max_g, z_max_g],
                [x_min_g, y_min_g, z_min_g],
                [x_max_g, y_min_g, z_min_g],
                [x_max_g, y_max_g, z_min_g],
                [x_min_g, y_max_g, z_min_g],
            ]
        )
        box_corners_manim = data.grid_to_manim(box_corners_grid)
        wireframe = _make_wireframe_box(box_corners_manim, SLAB_EDGE_COLOR)
        faces = _make_filled_faces(box_corners_manim, COLOR_SLAB, SLAB_OPACITY)
        slab_box = VGroup(faces, wireframe)

        # Assemble the reservoir group and scale/position for slot 0
        res_group = VGroup(perm_slices, well_lines, slab_box)
        res_group.scale(0.40)
        res_group.move_to(get_3d_pos(SCREEN_XS[0]))

        # ==================================================================
        # 2. CNN PRISMS (3-D, slot 1)
        # ==================================================================
        CNN_COLOR = ManimColor("#4fc3f7")  # light blue
        p1 = Prism(dimensions=[0.15, 1.4, 1.4]).set_color(CNN_COLOR).set_opacity(0.65)
        p2 = Prism(dimensions=[0.22, 1.0, 1.0]).set_color(CNN_COLOR).set_opacity(0.65)
        p3 = Prism(dimensions=[0.30, 0.6, 0.6]).set_color(CNN_COLOR).set_opacity(0.65)
        p4 = Prism(dimensions=[0.35, 0.3, 0.3]).set_color(CNN_COLOR).set_opacity(0.65)
        cnn_group = VGroup(p1, p2, p3, p4).arrange(RIGHT, buff=0.18)
        cnn_group.rotate(40 * DEGREES, axis=OUT)
        cnn_group.move_to(get_3d_pos(SCREEN_XS[1]))

        # ==================================================================
        # 3. FNN / MLP (3-D prisms, slot 2)
        # ==================================================================
        FNN_COLOR = ManimColor("#66bb6a")  # green
        fc1 = Prism(dimensions=[0.15, 1.3, 0.3]).set_color(FNN_COLOR).set_opacity(0.80)
        fc2 = Prism(dimensions=[0.15, 0.9, 0.3]).set_color(FNN_COLOR).set_opacity(0.80)
        fc3 = Prism(dimensions=[0.15, 0.5, 0.3]).set_color(FNN_COLOR).set_opacity(0.80)
        fnn_group = VGroup(fc1, fc2, fc3).arrange(RIGHT, buff=0.25)
        fnn_group.rotate(40 * DEGREES, axis=OUT)
        fnn_group.move_to(get_3d_pos(SCREEN_XS[2]))

        # ==================================================================
        # 4. EDGE EMBEDDING — graph snippet (2-D, slot 3, fixed-in-frame)
        # ==================================================================
        node_r = 0.32
        node_gap = 1.5  # vertical distance between nodes

        inj_node = Circle(
            radius=node_r,
            color=BLUE,
            fill_color=BLUE,
            fill_opacity=0.85,
            stroke_width=2.5,
        )
        ext_node = Circle(
            radius=node_r,
            color=ORANGE,
            fill_color=ORANGE,
            fill_opacity=0.85,
            stroke_width=2.5,
        )

        # Place nodes vertically centred at (0, 0)
        inj_node.move_to(UP * node_gap / 2)
        ext_node.move_to(DOWN * node_gap / 2)

        lbl_inj = Text("W11", font_size=18, color=WHITE).move_to(inj_node)
        lbl_ext = Text("W6", font_size=18, color=WHITE).move_to(ext_node)

        edge_line = Line(
            inj_node.get_center(),
            ext_node.get_center(),
            color=COLOR_INJ_EXT,
            stroke_width=5,
        )

        graph_group = VGroup(edge_line, inj_node, lbl_inj, ext_node, lbl_ext)
        graph_group.move_to(np.array([SCREEN_XS[3], 0.0, 0.0]))

        # ==================================================================
        # 5. ARROWS (2-D, fixed-in-frame)
        # ==================================================================
        arr_kwargs = dict(
            color=WHITE,
            stroke_width=3.5,
            buff=0.0,
            max_tip_length_to_length_ratio=0.25,
            max_stroke_width_to_length_ratio=4.0,
        )

        # Explicitly hardcode perfectly uniform arrows between the visual gaps.
        # Length of each arrow is exactly 1.0.
        # Arrow 1: right of Slab -> left of CNN
        arrow1 = Arrow(start=[-3.5, 0.0, 0], end=[-2.5, 0.0, 0], **arr_kwargs)
        # Arrow 2: right of CNN -> left of FNN
        arrow2 = Arrow(start=[-0.5, 0.0, 0], end=[0.5, 0.0, 0], **arr_kwargs)
        # Arrow 3: right of FNN -> left of Edge Embedding
        arrow3 = Arrow(start=[2.5, 0.0, 0], end=[3.5, 0.0, 0], **arr_kwargs)

        # We manually shift up all the 2D overlays to match the visual center of
        # the 3D items, which is Y=0.0 when mapped horizontally.
        arrows = VGroup(arrow1, arrow2, arrow3)
        arrows.shift(UP * COMP_Y)
        graph_group.shift(UP * COMP_Y)

        # ==================================================================
        # 6. LABELS (2-D, fixed-in-frame, below each component)
        # ==================================================================
        lbl_slab = Text("Physics Slab", font_size=24, color=WHITE)
        lbl_slab.move_to(np.array([SCREEN_XS[0], LABEL_Y, 0]))

        lbl_cnn = Text("3D CNN Encoder", font_size=24, color=WHITE)
        lbl_cnn.move_to(np.array([SCREEN_XS[1], LABEL_Y, 0]))

        lbl_fnn = Text("MLP", font_size=24, color=WHITE)
        lbl_fnn.move_to(np.array([SCREEN_XS[2], LABEL_Y, 0]))

        lbl_edge = Text("Edge Embedding", font_size=24, color=WHITE)
        lbl_edge.move_to(np.array([SCREEN_XS[3], LABEL_Y, 0]))

        # ==================================================================
        # ASSEMBLE
        # ==================================================================
        # Fixed-in-frame items: arrows, labels, graph
        fixed_2d = VGroup(
            arrows,
            lbl_slab,
            lbl_cnn,
            lbl_fnn,
            lbl_edge,
            graph_group,
        )

        # Camera
        self.set_camera_orientation(
            phi=75 * np.pi / 180,
            theta=-45 * np.pi / 180,
            zoom=GLOBAL_ZOOM,
        )

        # Add 3-D objects
        self.add(res_group, cnn_group, fnn_group)

        # Add 2-D overlays
        self.add_fixed_in_frame_mobjects(fixed_2d)

        self.wait(0.1)
