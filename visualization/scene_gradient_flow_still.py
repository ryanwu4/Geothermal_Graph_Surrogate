"""Gradient Flow Still.

An independent render showing the full pipeline with INVERTED YELLOW arrows
to represent backpropagation / gradient flow during optimization.
"""

from __future__ import annotations

import sys
from pathlib import Path
import random

import numpy as np
from manim import (
    BLUE,
    ORANGE,
    DOWN,
    RIGHT,
    UP,
    LEFT,
    WHITE,
    BLACK,
    YELLOW,
    ORIGIN,
    Arrow,
    Circle,
    Line,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Group,
    config,
    ManimColor,
    Axes,
    ImageMobject,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load  # noqa: E402

H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"
SLAB_IMG = (
    Path(__file__).resolve().parent
    / "media" / "images" / "scene_cnn_slab_still"
    / "CnnSlabStill_ManimCE_v0.20.1.png"
)

config.disable_caching = True
config.background_color = BLACK
config.save_last_frame = True
config.format = "png"

# Palettes (matched from scene_master)
COLOR_INJ_EXT = ManimColor("#00e5ff")
COLOR_INJ_INJ = ManimColor("#76ff03")
COLOR_EXT_EXT = ManimColor("#ea00ff")
COLOR_HI = ManimColor("#fde725")

class GradientFlowStill(Scene):
    def construct(self):
        data = load(H5_FILE)

        # Common NN box style matching exactly scene_master
        def nn_box(label: str, w=3.6, h=2.4):
            box = Rectangle(
                width=w,
                height=h,
                color=WHITE,
                fill_color=ManimColor("#222222"),
                fill_opacity=1,
            )
            text = Text(label, font_size=30).move_to(box.get_center())
            return VGroup(box, text)

        # -------------------------------------------------------------------
        # LINE 1: Edge Feature Extraction
        # -------------------------------------------------------------------

        # 1. Well Pair
        wp_node1 = Circle(radius=0.25, color=BLUE, fill_opacity=1.0)
        wp_node2 = Circle(radius=0.25, color=ORANGE, fill_opacity=1.0)
        wp_line = Line(wp_node1.get_right(), wp_node2.get_left(), color=COLOR_INJ_EXT, stroke_width=4)
        wp_nodes = VGroup(wp_node1, wp_line, wp_node2).arrange(RIGHT, buff=0.4)
        wp_text = Text("Well Pair (i, j)\n& 3D Distance", font_size=30)
        wp_group = VGroup(wp_text, wp_nodes).arrange(DOWN, buff=0.4)

        # 2. Physics Slab
        if SLAB_IMG.exists():
            slab_img = ImageMobject(str(SLAB_IMG))
            slab_img.height = 3.5
        else:
            slab_img = Rectangle(width=4.0, height=3.5, color=BLUE, fill_opacity=0.2)
        slab_text = Text("Physics Slab\n(8×16×32×32)", font_size=30)
        slab_group = Group(slab_text, slab_img).arrange(DOWN, buff=0.4)

        # 3. 3D CNN
        cnn_group = nn_box("3D CNN")

        # 4. MLP
        mlp_group = nn_box("MLP")

        # 5. Edge Feature
        feat_box = Rectangle(width=1.8, height=1.2, color=ManimColor("#fdcb6e"), fill_opacity=0.3)
        feat_text = Text("Edge Feature\nVector", font_size=30)
        feat_box_text = Text("e_ij", font_size=30, color=ManimColor("#fdcb6e")).move_to(feat_box)
        feat_group = VGroup(feat_text, VGroup(feat_box, feat_box_text)).arrange(DOWN, buff=0.4)

        # Increase buff to make lines longer, so arrows are properly sized
        row1 = Group(wp_group, slab_group, cnn_group, mlp_group, feat_group).arrange(RIGHT, buff=1.2)

        # -------------------------------------------------------------------
        # LINE 2: Graph Pipeline (from scene_master)
        # -------------------------------------------------------------------

        # 1. Well Graph (Target Graph Group)
        GRAPH_HALF = 4.32
        grid_xy = data.well_grid_coords[:, :2].astype(np.float64)
        cx, cy = grid_xy[:, 0].mean(), grid_xy[:, 1].mean()
        span = max(
            grid_xy[:, 0].max() - grid_xy[:, 0].min(),
            grid_xy[:, 1].max() - grid_xy[:, 1].min(),
            1.0,
        )
        scale_2d = 2 * GRAPH_HALF / span

        pos_2d = np.zeros((len(grid_xy), 3))
        pos_2d[:, 0] = (grid_xy[:, 1] - cy) * scale_2d
        pos_2d[:, 1] = (grid_xy[:, 0] - cx) * scale_2d

        node_circles = VGroup()
        node_labels = VGroup()
        for i in range(len(pos_2d)):
            col = BLUE if data.is_injector[i] else ORANGE
            pos_node = pos_2d[i]
            circ = Circle(radius=0.25, color=col, fill_opacity=1.0, stroke_width=2).move_to(pos_node)
            lbl = Text(f"W{i}", font_size=24, color=WHITE).move_to(pos_node)
            node_circles.add(circ)
            node_labels.add(lbl)

        straight_edges = VGroup()
        for pair in data.path_well_pairs:
            s, d = pair
            si, di = data.is_injector[s], data.is_injector[d]
            if si != di:
                col = COLOR_INJ_EXT
            elif si:
                col = COLOR_INJ_INJ
            else:
                col = COLOR_EXT_EXT
            straight_edges.add(Line(start=pos_2d[s], end=pos_2d[d], color=col, stroke_width=3))

        graph_group = VGroup(straight_edges, node_circles, node_labels)
        graph_group.scale(0.5)  # Scale to match pipeline

        graph_text = Text("Well Graph\n(with e_ij edges)", font_size=30)
        graph_full = VGroup(graph_text, graph_group).arrange(DOWN, buff=0.4)

        # 2. Heterogeneous GNN
        gnn_group = nn_box("Heterogeneous\nGNN")

        # 3. Message Passing Graph
        msg_graph_group = graph_group.copy()
        random.seed(42)
        for circ in msg_graph_group[1]:  # Index 1 is node_circles
            circ.set_color(
                random.choice([
                    ManimColor("#ff0044"), ManimColor("#00ff44"), ManimColor("#ddff00"),
                    ManimColor("#aa00ff"), ManimColor("#00eeff"), ManimColor("#ffaa00"),
                ])
            )
            circ.set_fill(opacity=1.0)
        msg_graph_label = Text("Message Passing\nGraph", font_size=30)
        msg_graph_full = VGroup(msg_graph_label, msg_graph_group).arrange(DOWN, buff=0.4)

        # 4. Aggregation Block
        agg_group = nn_box("Global Readout\n(Aggregation)")

        # 5. FNN Block
        fnn_group = nn_box("FNN")

        # 6. Prediction Output
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
        pred_group = VGroup(pred_label, plot_group).arrange(DOWN, buff=0.4)

        # Increase buff here too
        row2 = Group(
            graph_full, gnn_group, msg_graph_full,
            agg_group, fnn_group, pred_group
        ).arrange(RIGHT, buff=1.2)

        # -------------------------------------------------------------------
        # LAYOUT & REVERSED ARROWS
        # -------------------------------------------------------------------

        full_pipeline = Group(row1, row2).arrange(DOWN, buff=1.2)

        # Scale to fit standard screen wide
        if full_pipeline.width > 13.5:
            full_pipeline.scale(13.5 / full_pipeline.width)

        # Or if too tall
        if full_pipeline.height > 7.5:
            full_pipeline.scale(7.5 / full_pipeline.height)

        full_pipeline.move_to(ORIGIN)

        arr_kwargs = {
            "color": YELLOW,
            "stroke_width": 4,
            "buff": 0.05,
            "max_tip_length_to_length_ratio": 0.35,
            "max_stroke_width_to_length_ratio": 5.0
        }

        # Draw arrows for Row 1 (Reversed)
        arrows1 = VGroup()
        for i in range(len(row1) - 1):
            arr = Arrow(
                start=row1[i + 1].get_left(),
                end=row1[i].get_right(),
                **arr_kwargs
            )
            arrows1.add(arr)

        # Draw arrows for Row 2 (Reversed)
        arrows2 = VGroup()
        for i in range(len(row2) - 1):
            arr = Arrow(
                start=row2[i + 1].get_left(),
                end=row2[i].get_right(),
                **arr_kwargs
            )
            arrows2.add(arr)

        # Connecting arrow from Well Graph to Edge Feature (Reversed)
        # Gradient runs UP from row2[0] to row1[-1]
        pt1 = row1[-1].get_bottom() + DOWN * 0.05
        mid_y = full_pipeline.get_center()[1]
        pt2 = np.array([pt1[0], mid_y, 0])
        pt3 = np.array([row2[0].get_top()[0], mid_y, 0])
        pt4 = row2[0].get_top() + UP * 0.05

        # Path: pt4 -> pt3 -> pt2 -> pt1
        l1 = Line(pt4, pt3, color=YELLOW, stroke_width=4)
        l2 = Line(pt3, pt2, color=YELLOW, stroke_width=4)
        a3 = Arrow(pt2, pt1, color=YELLOW, stroke_width=4, buff=0.0, max_tip_length_to_length_ratio=0.3)
        connect_arrow = VGroup(l1, l2, a3)

        self.add(full_pipeline, arrows1, arrows2, connect_arrow)
