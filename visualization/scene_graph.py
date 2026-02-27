"""Scene 4 — Path-to-Graph Conversion (2-D, Type-Coloured).

Starts with all A* paths and well nodes displayed. Animates:
  1. Paths morph from 3-D polylines into straight edges.
  2. Edges coloured by connection type.
  3. 90-degree rotated layout. No headers.
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
    Circle,
    Create,
    FadeIn,
    Line,
    Scene,
    Text,
    Transform,
    VGroup,
    config,
    ManimColor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
# v2.5.1: Maintain 2D zoom
GRAPH_HALF = 4.32
NODE_RADIUS = 0.25
NODE_LABEL_SIZE = 18

config.disable_caching = True

# Connection Type Colours
COLOR_INJ_EXT = ManimColor("#00e5ff")  # Cyan
COLOR_INJ_INJ = ManimColor("#76ff03")  # Lime
COLOR_EXT_EXT = ManimColor("#ea00ff")  # Magenta


class GraphScene(Scene):
    """2-D flat graph scene (Rotated 90)."""

    def construct(self):
        data = load(H5_FILE)

        # --- 2-D well positions --------------------------------------------
        grid_xy = data.well_grid_coords[:, :2].astype(np.float64)
        cx, cy = grid_xy[:, 0].mean(), grid_xy[:, 1].mean()

        span_x = grid_xy[:, 0].max() - grid_xy[:, 0].min()
        span_y = grid_xy[:, 1].max() - grid_xy[:, 1].min()
        span = max(span_x, span_y, 1.0)
        scale_2d = 2 * GRAPH_HALF / span

        pos_2d = np.zeros((len(grid_xy), 3))
        # v2.5.1: Flip X and Y to effect a 90-degree rotation
        pos_2d[:, 0] = (grid_xy[:, 1] - cy) * scale_2d
        pos_2d[:, 1] = (grid_xy[:, 0] - cx) * scale_2d

        # --- Nodes ---------------------------------------------------------
        node_mobs = VGroup()
        labels = VGroup()
        for i in range(len(pos_2d)):
            col = BLUE if data.is_injector[i] else ORANGE
            circ = Circle(
                radius=NODE_RADIUS, color=col, fill_opacity=0.85, stroke_width=2
            )
            circ.move_to(pos_2d[i])
            lbl = Text(f"W{i}", font_size=NODE_LABEL_SIZE, color=WHITE)
            lbl.move_to(pos_2d[i])
            node_mobs.add(circ)
            labels.add(lbl)

        # --- A* Paths (projected) ------------------------------------------
        path_mobs: list[VGroup] = []
        for k, (path_3d, pair) in enumerate(zip(data.paths, data.path_well_pairs)):
            src_idx, dst_idx = pair
            si, di = data.is_injector[src_idx], data.is_injector[dst_idx]
            if si != di:
                col = COLOR_INJ_EXT
            elif si:
                col = COLOR_INJ_INJ
            else:
                col = COLOR_EXT_EXT

            # Map to 2D with 90-deg rotation (Swap g_x and g_y)
            g_x = (path_3d[:, 0] / data.scale) + data.offset[0]
            g_y = (path_3d[:, 1] / data.scale) + data.offset[1]

            p_rot = np.zeros_like(path_3d)
            p_rot[:, 0] = (g_y - cy) * scale_2d
            p_rot[:, 1] = (g_x - cx) * scale_2d

            segs = VGroup()
            for j in range(len(p_rot) - 1):
                seg = Line(start=p_rot[j], end=p_rot[j + 1], color=col, stroke_width=2)
                segs.add(seg)
            path_mobs.append(segs)

        # --- Animation -----------------------------------------------------
        # v2.5.1: Nodes present from start
        self.add(node_mobs, labels)
        self.play(FadeIn(node_mobs), FadeIn(labels), run_time=0.5)

        # Show initial paths
        self.play(*(Create(pm, run_time=1.0) for pm in path_mobs))
        self.wait(1.5)

        # Morph paths to straight edges
        morph_anims = []
        for k, (pm, pair) in enumerate(zip(path_mobs, data.path_well_pairs)):
            src, dst = pair
            si, di = data.is_injector[src], data.is_injector[dst]
            if si != di:
                col = COLOR_INJ_EXT
            elif si:
                col = COLOR_INJ_INJ
            else:
                col = COLOR_EXT_EXT

            straight = Line(
                start=pos_2d[src], end=pos_2d[dst], color=col, stroke_width=3
            )
            morph_anims.append(Transform(pm, straight))

        self.play(*morph_anims, run_time=2.0)

        # --- Legend ---
        leg_ie = VGroup(
            Line(ORIGIN, 0.4 * RIGHT, color=COLOR_INJ_EXT, stroke_width=4),
            Text("Inj-Ext", font_size=16),
        ).arrange(RIGHT)
        leg_ii = VGroup(
            Line(ORIGIN, 0.4 * RIGHT, color=COLOR_INJ_INJ, stroke_width=4),
            Text("Inj-Inj", font_size=16),
        ).arrange(RIGHT)
        leg_ee = VGroup(
            Line(ORIGIN, 0.4 * RIGHT, color=COLOR_EXT_EXT, stroke_width=4),
            Text("Ext-Ext", font_size=16),
        ).arrange(RIGHT)
        legend = (
            VGroup(leg_ie, leg_ii, leg_ee)
            .arrange(DOWN, buff=0.25, aligned_edge=LEFT)
            .to_corner(DOWN + RIGHT, buff=0.5)
        )

        self.play(FadeIn(legend))
        self.wait(4)
