"""Dedicated still: the gated seed-1042 best well configuration in one geology.

Static single-frame companion to scene_config_geology_diversity — same mesh,
wells, and palette machinery (imported from that module), with two deliberate
differences:

  * camera sits at a slightly higher angle (PHI_DEG = 60 vs the video's 75),
  * the permeability color scale is fit to THIS geology's own [2, 98]
    percentile range instead of the 4-geology pooled range, so the full
    colormap span is spent on the rock actually shown.

Render (from this directory, geothermal-pomdp env):
  manim -ql -s scene_best_config_still.py BestConfigStill   # fast check
  manim -qh -s scene_best_config_still.py BestConfigStill   # 1080p still
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    Rectangle,
    Text,
    ThreeDScene,
    VGroup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene_config_geology_diversity import (  # noqa: E402
    COLOR_INJ,
    COLOR_PROD,
    CONFIGS,
    GridMap,
    build_perm_slices,
    build_well_lines,
    geology_files,
    load_config_wells,
    load_geology,
    make_colorbar,
)

# ---------------------------------------------------------------------------
GEO_INDEX = 0          # geology realization to show (0 = scenario 71, v2.5_0010)
CONFIG = CONFIGS[0]    # (iter 4, run000000) = the strict-EMV best config
PHI_DEG = 60.0         # camera elevation; the flyaround uses 75 (lower = more top-down)
THETA_DEG = -45.0
ZOOM = 1.0


class BestConfigStill(ThreeDScene):
    def construct(self):
        logp, valid = load_geology(geology_files()[GEO_INDEX])
        wells = load_config_wells(*CONFIG)

        nz, nx, ny = logp.shape
        gm = GridMap(nx=nx, ny=ny, nz=nz)
        depths = np.linspace(0, nz - 1, 6).astype(int)

        # color scale cropped to THIS geology's own range
        p_lo, p_hi = np.percentile(logp[valid], [2, 98])

        slices = build_perm_slices(logp, valid, gm, depths, p_lo, p_hi)
        well_lines = build_well_lines(wells, gm)

        self.set_camera_orientation(phi=PHI_DEG * np.pi / 180,
                                    theta=THETA_DEG * np.pi / 180, zoom=ZOOM)
        colorbar = make_colorbar(p_lo, p_hi)
        self.add_fixed_in_frame_mobjects(colorbar)
        colorbar.to_edge(RIGHT, buff=0.5)

        # well-type legend, text style matched to the colorbar labels
        def _entry(color, label):
            swatch = Rectangle(width=0.45, height=0.12, stroke_width=0,
                               fill_color=color, fill_opacity=1.0)
            return VGroup(swatch, Text(label, font_size=24)).arrange(
                RIGHT, buff=0.18)

        legend = VGroup(
            _entry(COLOR_INJ, "Injector"),
            _entry(COLOR_PROD, "Producer"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        self.add_fixed_in_frame_mobjects(legend)
        legend.to_corner(UP + LEFT, buff=0.6)

        self.add(slices, well_lines)
        # one trivial animation so `manim -s` has a frame to save
        self.play(FadeIn(slices, run_time=0.1))
