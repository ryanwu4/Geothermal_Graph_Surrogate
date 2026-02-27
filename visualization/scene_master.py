"""Scene Master — Production Blend.

A single continuous sequence:
1. Fault Structure (orange slices).
2. Blends into Permeability heatmap.
3. Blends into A* Paths.
4. Morphs into 2D Graph (rotated 90).

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
    FadeOut,
    Line,
    Line3D,
    Rectangle,
    Scene,
    Square,
    Text,
    ThreeDScene,
    Transform,
    VGroup,
    config,
    interpolate_color,
    ManimColor,
    Arrow,
    Axes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load  # noqa: E402

# ---------------------------------------------------------------------------
H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
SLICE_RESOLUTON = 80
SLICE_COUNT_FAULT = 10
SLICE_COUNT_PERM = 7
SLICE_OPACITY_PERM = 0.72
SLICE_OPACITY_ASTAR = 0.28
WELL_THICKNESS = 0.04
PATH_WIDTH = 0.025
ROTATION_RATE = 0.08
GLOBAL_ZOOM = 1.0

config.disable_caching = True

# Palettes
FAULT_COLOR = ManimColor("#ff5722")
COLOR_LO, COLOR_MID, COLOR_HI = (
    ManimColor("#440154"),
    ManimColor("#21918c"),
    ManimColor("#fde725"),
)

# v2.8.1: A* Visibility (RED)
COLOR_ASTAR = ManimColor("#ff0000")

# v2.8.1: Original Graph Colours
COLOR_INJ_EXT = ManimColor("#00e5ff")  # Cyan
COLOR_INJ_INJ = ManimColor("#76ff03")  # Lime
COLOR_EXT_EXT = ManimColor("#ea00ff")  # Magenta


def _perm_color(t: float) -> ManimColor:
    t = np.clip(t, 0.0, 1.0)
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2.0)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2.0)


def _make_colorbar(p_lo: float, p_hi: float) -> VGroup:
    bar = Rectangle(height=3, width=0.2, stroke_color=WHITE, stroke_width=1)
    bar.set_fill([COLOR_LO, COLOR_MID, COLOR_HI], opacity=1)
    t_min = Text(f"{p_lo:.1f}", font_size=24).next_to(bar, DOWN, buff=0.1)
    t_max = Text(f"{p_hi:.1f}", font_size=24).next_to(bar, UP, buff=0.1)
    t_label = Text("log10(k)", font_size=20).next_to(t_max, UP, buff=0.2)
    cb = VGroup(bar, t_min, t_max, t_label)
    return cb


class MasterScene(ThreeDScene):
    def construct(self):
        data = load(H5_FILE)

        # 1. PREPARE COMMON MOJECTS -----------------------------------------

        # Wells
        well_lines = VGroup()
        for i in range(len(data.well_tops)):
            line = Line3D(
                start=data.well_tops[i],
                end=data.well_bottoms[i],
                color=BLUE if data.is_injector[i] else ORANGE,
                thickness=WELL_THICKNESS,
            )
            well_lines.add(line)

        # 2. STATE 1: FAULTS ------------------------------------------------

        nz, nx, ny = data.grid_shape
        start_z_f = int(data.well_top_grid[:, 2].max()) + 20
        depths_f = np.linspace(start_z_f, nz - 1, SLICE_COUNT_FAULT).astype(int)
        xi, yi = np.linspace(0, nx - 1, SLICE_RESOLUTON).astype(int), np.linspace(
            0, ny - 1, SLICE_RESOLUTON
        ).astype(int)
        sq_size = max(nx / SLICE_RESOLUTON, ny / SLICE_RESOLUTON) * data.scale * 1.05

        fault_slices = VGroup()
        for zl in depths_f:
            layer = VGroup()
            for i in xi:
                for j in yi:
                    if not data.valid_mask[zl, i, j]:
                        pos = data.grid_to_manim(np.array([[i, j, zl]]))[0]
                        sq = Square(
                            side_length=sq_size,
                            stroke_width=0,
                            fill_color=FAULT_COLOR,
                            fill_opacity=0.5,
                        )
                        sq.move_to(pos)
                        layer.add(sq)
            fault_slices.add(layer)

        self.set_camera_orientation(
            phi=75 * np.pi / 180, theta=-45 * np.pi / 180, zoom=GLOBAL_ZOOM
        )

        group = VGroup(fault_slices, well_lines)
        self.add(group)
        title_faults = Text("Fault Structure & Wells", font_size=48).to_corner(
            UP + LEFT
        )
        self.add_fixed_in_frame_mobjects(title_faults)

        self.play(FadeIn(group), FadeIn(title_faults), run_time=2)
        self.begin_ambient_camera_rotation(rate=ROTATION_RATE)
        self.wait(1)
        self.play(FadeOut(title_faults))
        self.wait(2)

        # 3. TRANSITION TO PERMEABILITY -------------------------------------

        log_perm_total = np.log10(np.clip(data.perm[data.valid_mask], 1e-18, None))
        p_lo, p_hi = (
            np.percentile(log_perm_total, [2, 98])
            if len(log_perm_total) > 0
            else (-16, -10)
        )
        depths_p = np.linspace(0, nz - 1, SLICE_COUNT_PERM).astype(int)

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
                        )
                        sq.move_to(pos)
                        layer.add(sq)
            perm_slices.add(layer)

        title_perm = Text("Permeability", font_size=60).to_corner(UP + LEFT)
        self.add_fixed_in_frame_mobjects(title_perm)

        colorbar = _make_colorbar(p_lo, p_hi)
        self.add_fixed_in_frame_mobjects(colorbar)
        colorbar.to_edge(RIGHT, buff=0.5)

        self.play(
            Transform(fault_slices, perm_slices),
            FadeIn(title_perm),
            FadeIn(colorbar),
            run_time=3,
        )
        self.wait(1)
        self.play(FadeOut(title_perm))
        self.wait(2)

        # 4. TRANSITION TO A* PATHS -----------------------------------------

        title_astar = Text("A* Paths", font_size=60).to_corner(UP + LEFT)
        self.add_fixed_in_frame_mobjects(title_astar)

        self.play(
            fault_slices.animate.set_style(fill_opacity=SLICE_OPACITY_ASTAR),
            FadeIn(title_astar),
            run_time=2,
        )

        path_mobs = VGroup()
        for k, (path, pair) in enumerate(zip(data.paths, data.path_well_pairs)):
            # v2.8.1: A* paths use RED for stand-out visibility in 3D
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

            self.play(Create(polyline, run_time=0.4))
            path_mobs.add(polyline)

        self.wait(1)
        self.play(FadeOut(title_astar))
        self.wait(2)
        self.stop_ambient_camera_rotation()

        # 5. TRANSITION TO 2-D GRAPH ----------------------------------------

        GRAPH_HALF = 4.32
        grid_xy = data.well_grid_coords[:, :2].astype(np.float64)
        cx, cy = grid_xy[:, 0].mean(), grid_xy[:, 1].mean()
        span = max(
            grid_xy[:, 0].max() - grid_xy[:, 0].min(),
            grid_xy[:, 1].max() - grid_xy[:, 1].min(),
            1.0,
        )
        scale_2d = 2 * GRAPH_HALF / span

        # 90-degree rotated positions
        pos_2d = np.zeros((len(grid_xy), 3))
        pos_2d[:, 0] = (grid_xy[:, 1] - cy) * scale_2d
        pos_2d[:, 1] = (grid_xy[:, 0] - cx) * scale_2d

        node_circles = VGroup()
        node_labels = VGroup()
        for i in range(len(pos_2d)):
            col = BLUE if data.is_injector[i] else ORANGE
            # add positive z offset so nodes appear in front
            pos_node = pos_2d[i] + np.array([0, 0, 0.05])
            circ = (
                Circle(radius=0.25, color=col, fill_opacity=1.0, stroke_width=2)
                .move_to(pos_node)
                .set_z_index(1)
            )
            lbl = (
                Text(f"W{i}", font_size=24, color=WHITE)
                .move_to(pos_node)
                .set_z_index(2)
            )
            node_circles.add(circ)
            node_labels.add(lbl)

        straight_edges = VGroup()
        for k, pair in enumerate(data.path_well_pairs):
            s, d = pair
            si, di = data.is_injector[s], data.is_injector[d]
            # v2.8.1: Revert 2D graph connections to connection-type colors
            if si != di:
                col = COLOR_INJ_EXT
            elif si:
                col = COLOR_INJ_INJ
            else:
                col = COLOR_EXT_EXT

            straight_edges.add(
                Line(
                    start=pos_2d[s], end=pos_2d[d], color=col, stroke_width=3
                ).set_z_index(0)
            )

        self.move_camera(
            phi=0,
            theta=-90 * np.pi / 180,
            added_anims=[FadeOut(fault_slices), FadeOut(colorbar)],
            run_time=3,
        )
        self.play(
            Transform(path_mobs, straight_edges),
            Transform(well_lines, node_circles),
            FadeIn(node_labels),
            run_time=3,
        )

        leg_ie = VGroup(
            Line(ORIGIN, 0.4 * RIGHT, color=COLOR_INJ_EXT),
            Text("Inj-Ext", font_size=36),
        ).arrange(RIGHT)
        leg_ii = VGroup(
            Line(ORIGIN, 0.4 * RIGHT, color=COLOR_INJ_INJ),
            Text("Inj-Inj", font_size=36),
        ).arrange(RIGHT)
        leg_ee = VGroup(
            Line(ORIGIN, 0.4 * RIGHT, color=COLOR_EXT_EXT),
            Text("Ext-Ext", font_size=36),
        ).arrange(RIGHT)
        legend = (
            VGroup(leg_ie, leg_ii, leg_ee)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_corner(DOWN + RIGHT)
        )
        self.play(FadeIn(legend))

        self.wait(2)

        # 6. TRANSITION TO GNN PIPELINE -------------------------------------

        # Ensure the older transformed objects are cleanly replaced by the actual VGroup
        # to prevent background replicas during the pipeline layout
        self.remove(path_mobs, well_lines, node_labels)
        graph_group = VGroup(straight_edges, node_circles, node_labels)
        self.add(graph_group)

        # Create target copy for graph to be part of the pipeline layout - halved in size
        target_graph_group = graph_group.copy().scale(0.5)

        # GNN Block
        gnn_box = Rectangle(
            width=3.6,
            height=2.4,
            color=WHITE,
            fill_color=ManimColor("#222222"),
            fill_opacity=1,
        )
        gnn_text = Text("Heterogeneous\nGNN", font_size=30).move_to(
            gnn_box.get_center()
        )
        gnn_group = VGroup(gnn_box, gnn_text)

        # Message Passing Graph
        msg_graph_group = target_graph_group.copy()
        import random

        random.seed(42)
        # msg_graph_group[0] are the node_circles
        for circ in msg_graph_group[0]:
            circ.set_color(
                random.choice(
                    [
                        ManimColor("#ff0044"),
                        ManimColor("#00ff44"),
                        ManimColor("#ddff00"),
                        ManimColor("#aa00ff"),
                        ManimColor("#00eeff"),
                        ManimColor("#ffaa00"),
                    ]
                )
            )
            circ.set_fill(opacity=1.0)
        msg_graph_label = Text("Message Passing\nGraph", font_size=30)
        msg_graph_full = VGroup(msg_graph_group, msg_graph_label).arrange(
            DOWN, buff=0.4
        )

        # Aggregation Block
        agg_box = Rectangle(
            width=3.6,
            height=2.4,
            color=WHITE,
            fill_color=ManimColor("#222222"),
            fill_opacity=1,
        )
        agg_text = Text("Global Readout\n(Aggregation)", font_size=30).move_to(
            agg_box.get_center()
        )
        agg_group = VGroup(agg_box, agg_text)

        # FNN Block
        fnn_box = Rectangle(
            width=3.6,
            height=2.4,
            color=WHITE,
            fill_color=ManimColor("#222222"),
            fill_opacity=1,
        )
        fnn_text = Text("FNN", font_size=30).move_to(fnn_box.get_center())
        fnn_group = VGroup(fnn_box, fnn_text)

        # Prediction Output
        pred_label = Text("Total Energy\nProduction", font_size=30)
        ax = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=1.5,
            y_length=1.2,
            axis_config={"include_numbers": False, "include_tip": False},
        )
        curve = ax.plot(lambda x: 8 * (1 - np.exp(-x / 3)), color=COLOR_HI)
        plot_group = VGroup(ax, curve)
        pred_group = VGroup(pred_label, plot_group).arrange(DOWN, buff=0.2)

        # Now arrange the sequence
        pipeline_seq = VGroup(
            target_graph_group,
            gnn_group,
            msg_graph_full,
            agg_group,
            fnn_group,
            pred_group,
        ).arrange(RIGHT, buff=0.45)

        # Scale down if it's too wide
        if pipeline_seq.width > 13.5:
            pipeline_seq.scale(13.5 / pipeline_seq.width)

        # Re-center
        pipeline_seq.move_to(ORIGIN)

        # Create arrows based on new positions
        arrows = VGroup()
        for i in range(len(pipeline_seq) - 1):
            start_obj = pipeline_seq[i]
            end_obj = pipeline_seq[i + 1]
            arr = Arrow(
                start=start_obj.get_right(),
                end=end_obj.get_left(),
                buff=0.15,
                color=WHITE,
                stroke_width=4,
            )
            arrows.add(arr)

        # Animate
        self.play(FadeOut(legend))

        # Shrink and move the original graph
        self.play(Transform(graph_group, target_graph_group), run_time=1.5)

        # Show pipeline step-by-step
        self.play(FadeIn(gnn_group), Create(arrows[0]), run_time=0.8)
        self.play(FadeIn(msg_graph_full), Create(arrows[1]), run_time=0.8)
        self.play(FadeIn(agg_group), Create(arrows[2]), run_time=0.8)
        self.play(FadeIn(fnn_group), Create(arrows[3]), run_time=0.8)
        self.play(FadeIn(pred_group), Create(arrows[4]), run_time=0.8)

        self.wait(4)
