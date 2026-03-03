"""Scene for visualizing 2D Path-to-Graph with Node and Edge features.

Outputs two static images:
  1. Nodes labeled with Injection rate, Depth, Perm, Poro, Press, Temp
  2. Edges labeled with length, P, T, poro, perm, TOF
"""

import sys
from pathlib import Path
import h5py
import numpy as np

from manim import (
    BLUE, DOWN, LEFT, ORANGE, ORIGIN, RIGHT, UP, WHITE, Circle, Create,
    FadeIn, Line, Scene, Text, Transform, VGroup, config, ManimColor,
    Rectangle, Axes
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReservoirData, load

H5_FILE = Path(__file__).resolve().parent.parent / "data_test" / "v2.5_0111.h5"

# Let's see if H5_FILE exists. If not, fallback to data/
if not H5_FILE.exists():
    import glob
    files = glob.glob(str(Path(__file__).resolve().parent.parent / "data" / "v2.5_*.h5"))
    if files:
        H5_FILE = Path(files[0])


GRAPH_HALF_W = 6.0
GRAPH_HALF_H = 3.5
NODE_RADIUS = 0.25
NODE_LABEL_SIZE = 12
EDGE_LABEL_SIZE = 10

config.disable_caching = True
config.pixel_width = 1920
config.pixel_height = 1080

# Connection Type Colours
COLOR_INJ_EXT = ManimColor("#00e5ff")  # Cyan
COLOR_INJ_INJ = ManimColor("#76ff03")  # Lime
COLOR_EXT_EXT = ManimColor("#ea00ff")  # Magenta

class NodeFeatureGraph(Scene):
    def construct(self):
        data = load(H5_FILE)

        # We also need injection rates which aren't in data_loader's ReservoirData
        # Let's extract node features directly from raw grid.
        n_wells = len(data.well_tops)
        w_x = data.well_top_grid[:, 0].astype(int)
        w_y = data.well_top_grid[:, 1].astype(int)
        w_min_z = data.well_top_grid[:, 2].astype(int)
        w_max_z = data.well_bottom_grid[:, 2].astype(int)
        
        node_stats = []
        with h5py.File(H5_FILE, "r") as src:
            inj_rate_grid = src["Input/InjRate"][:]
        
        for i in range(n_wells):
            z_slice = slice(w_min_z[i], w_max_z[i] + 1)
            # mean over profile
            if w_min_z[i] == w_max_z[i]:
                perm_mean = data.perm[w_min_z[i], w_x[i], w_y[i]]
                poro_mean = data.porosity[w_min_z[i], w_x[i], w_y[i]]
                temp_mean = data.temp[w_min_z[i], w_x[i], w_y[i]]
                press_mean = data.press[w_min_z[i], w_x[i], w_y[i]]
            else:
                perm_mean = np.mean(data.perm[z_slice, w_x[i], w_y[i]])
                poro_mean = np.mean(data.porosity[z_slice, w_x[i], w_y[i]])
                temp_mean = np.mean(data.temp[z_slice, w_x[i], w_y[i]])
                press_mean = np.mean(data.press[z_slice, w_x[i], w_y[i]])
            
            w_inj_rate = inj_rate_grid[w_max_z[i], w_x[i], w_y[i]]
            depth = np.abs(w_max_z[i] - w_min_z[i]) * 10.0 # approximation or just use layers?
            depth_str = f"L{w_min_z[i]}-{w_max_z[i]}"

            stat_str = (
                f"Inj={w_inj_rate:.1f}\n"
                f"D={depth_str}\n" 
                f"K={perm_mean:.1e}\n"
                f"Phi={poro_mean:.2f}\n"
                f"P={press_mean/1e6:.1f}M\n"
                f"T={temp_mean:.1f}"
            )
            node_stats.append(stat_str)

        grid_xy = data.well_grid_coords[:, :2].astype(np.float64)
        cx, cy = grid_xy[:, 0].mean(), grid_xy[:, 1].mean()
        span_x = grid_xy[:, 0].max() - grid_xy[:, 0].min()
        span_y = grid_xy[:, 1].max() - grid_xy[:, 1].min()
        
        # Scale to fill screen nicely
        scale_x = (2 * GRAPH_HALF_W) / max(span_x, 1.0)
        scale_y = (2 * GRAPH_HALF_H) / max(span_y, 1.0)

        pos_2d = np.zeros((n_wells, 3))
        pos_2d[:, 0] = (grid_xy[:, 0] - cx) * scale_x
        pos_2d[:, 1] = (grid_xy[:, 1] - cy) * scale_y

        # Draw nodes
        node_mobs = VGroup()
        for i in range(n_wells):
            col = BLUE if data.is_injector[i] else ORANGE
            circ = Circle(radius=NODE_RADIUS, color=col, fill_opacity=0.85, stroke_width=2)
            circ.move_to(pos_2d[i])
            
            # Label
            lbl = Text(f"W{i}", font_size=NODE_LABEL_SIZE, color=WHITE)
            lbl.move_to(pos_2d[i] + UP * 0.4)
            
            # Stats info box
            stats_text = Text(node_stats[i], font_size=NODE_LABEL_SIZE-2, color=WHITE, line_spacing=0.8)
            stats_text.next_to(circ, RIGHT if i % 2 == 0 else LEFT, buff=0.1)
            
            # Tiny bg for text readability
            bg = Rectangle(width=stats_text.width + 0.1, height=stats_text.height + 0.1, color=WHITE, stroke_width=0, fill_color='#222222', fill_opacity=0.7)
            bg.move_to(stats_text)
            
            node_mobs.add(VGroup(circ, lbl, bg, stats_text))

        # Draw edges (just lines, no text since this is node scene)
        edge_mobs = VGroup()
        for k, pair in enumerate(data.path_well_pairs):
            src, dst = pair
            si, di = data.is_injector[src], data.is_injector[dst]
            if si != di: col = COLOR_INJ_EXT
            elif si: col = COLOR_INJ_INJ
            else: col = COLOR_EXT_EXT
            
            line = Line(start=pos_2d[src], end=pos_2d[dst], color=col, stroke_width=2, stroke_opacity=0.4)
            edge_mobs.add(line)

        self.add(edge_mobs, node_mobs)
        
        # Add legend
        title = Text("Node Features (Profile Averaged)", font_size=24, color=WHITE).to_corner(UP + LEFT)
        self.add(title)


class EdgeFeatureGraph(Scene):
    def construct(self):
        data = load(H5_FILE)

        grid_xy = data.well_grid_coords[:, :2].astype(np.float64)
        cx, cy = grid_xy[:, 0].mean(), grid_xy[:, 1].mean()
        span_x = grid_xy[:, 0].max() - grid_xy[:, 0].min()
        span_y = grid_xy[:, 1].max() - grid_xy[:, 1].min()
        
        scale_x = (2 * GRAPH_HALF_W) / max(span_x, 1.0)
        scale_y = (2 * GRAPH_HALF_H) / max(span_y, 1.0)

        pos_2d = np.zeros((len(grid_xy), 3))
        pos_2d[:, 0] = (grid_xy[:, 0] - cx) * scale_x
        pos_2d[:, 1] = (grid_xy[:, 1] - cy) * scale_y

        node_mobs = VGroup()
        for i in range(len(pos_2d)):
            col = BLUE if data.is_injector[i] else ORANGE
            circ = Circle(radius=NODE_RADIUS, color=col, fill_opacity=0.85, stroke_width=2)
            circ.move_to(pos_2d[i])
            node_mobs.add(circ)

        edge_mobs = VGroup()
        edge_labels = VGroup()
        
        for k, pair in enumerate(data.path_well_pairs):
            src, dst = pair
            si, di = data.is_injector[src], data.is_injector[dst]
            if si != di: col = COLOR_INJ_EXT
            elif si: col = COLOR_INJ_INJ
            else: col = COLOR_EXT_EXT
            
            line = Line(start=pos_2d[src], end=pos_2d[dst], color=col, stroke_width=3, stroke_opacity=0.7)
            edge_mobs.add(line)
            
            # Edge stats from A* geology attributes
            # length(0), TOF(1), perm(4), poro(7), minT(12), minP(13)
            if data.edge_attr.shape[0] > k:
                attr = data.edge_attr[k]
                plen = attr[0]
                tof = attr[1]
                perm = attr[4]
                poro = attr[7]
                t = attr[12]
                p = attr[13]

                label_str = (
                    f"L={plen:.0f}\n"
                    f"TOF={tof/1e7:.1f}e7\n"
                    f"K={perm:.1e}\n"
                    f"Phi={poro:.2f}\n"
                    f"T={t:.1f} P={p/1e6:.1f}M"
                )
                
                txt = Text(label_str, font_size=EDGE_LABEL_SIZE, color=WHITE, line_spacing=0.8)
                # place text on middle of the line
                txt.move_to(line.get_center() + UP * 0.1)
                
                bg = Rectangle(width=txt.width + 0.1, height=txt.height + 0.1, color=WHITE, stroke_width=0, fill_color='#111111', fill_opacity=0.75)
                bg.move_to(txt)
                
                edge_labels.add(VGroup(bg, txt))

        self.add(edge_mobs, node_mobs, edge_labels)
        
        # Title
        title = Text("Edge Features (Extracted from A* Paths)", font_size=24, color=WHITE).to_corner(UP + LEFT)
        self.add(title)
