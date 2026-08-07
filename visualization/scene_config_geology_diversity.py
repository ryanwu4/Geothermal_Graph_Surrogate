"""Config/geology diversity flyaround.

One continuous ambient-rotation shot in the style of MasterScene's opening:
3D permeability slice stack + vertical well lines. While the camera orbits:

  Phase 1 — the SAME well configuration is shown while the geology morphs
            through 4 ensemble realizations (same wells, different rock).
  Phase 2 — the SAME geology stays while the well configuration morphs
            through 4 CMA-multistart exploit candidates (same rock,
            different wells).

Data: geology realizations from al_local_data/geology_h5s (the 15-member
ensemble, via configs/geologies_full_local.json), well configurations from the
gated seed-1042 run's exploit snapshot JSONs (acquire/iter_*/snapshots_json).
Well (x, y) indexes PermX[k, x, y] per the grid-coordinate audit; wells are
drawn from the surface down to their perforation depth z.

Render (from this directory, geothermal-pomdp env):
  manim -ql -s scene_config_geology_diversity.py ConfigGeologyDiversity  # last frame
  manim -qh scene_config_geology_diversity.py ConfigGeologyDiversity    # full 1080p
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    WHITE,
    FadeIn,
    Line3D,
    ManimColor,
    Rectangle,
    Square,
    Text,
    ThreeDScene,
    Transform,
    VGroup,
    config,
    interpolate_color,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
AL_REPO = Path("/home/rwu4/omv_geothermal/geothermal_active_learning")
RUN_ROOT = AL_REPO / "local_workspace_cma_npv_multistart_depth5055_gated_seed1042"
GEOLOGIES_CFG = AL_REPO / "configs" / "geologies_full_local.json"

# 4 ensemble realizations (geology_index into geologies_full_local.json);
# includes geology 8, the tight OOD reservoir.
GEO_INDICES = [0, 5, 8, 12]
# 4 exploit candidates from the gated seed-1042 run: (acquire iter, snapshot id
# suffix). First is the strict-EMV best config (iter 4, 79.3 M$ ensemble NPV).
CONFIGS = [
    (4, "run000000_step0040_exploit"),
    (1, "run000019_step0040_exploit"),
    (12, "run000002_step0040_exploit"),
    (19, "run000001_step0040_exploit"),
]

MAX_Z = 70               # clip depth: deepest perforation (~55) + margin
SLICE_COUNT = 6
SLICE_OPACITY = 0.72
WELL_THICKNESS = 0.05
ROTATION_RATE = 0.08
MANIM_HALF_WIDTH = 3.0

config.disable_caching = True

# figstyle dark-theme categorical colors (injector = blue, producer = orange)
COLOR_INJ = ManimColor("#58C4DD")
COLOR_PROD = ManimColor("#FF9000")
COLOR_LO, COLOR_MID, COLOR_HI = (
    ManimColor("#440154"),
    ManimColor("#21918c"),
    ManimColor("#fde725"),
)


def _perm_color(t: float) -> ManimColor:
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return interpolate_color(COLOR_LO, COLOR_MID, t * 2.0)
    return interpolate_color(COLOR_MID, COLOR_HI, (t - 0.5) * 2.0)


# ---------------------------------------------------------------------------
# Data loading (lightweight: geology cubes + snapshot well configs)
# ---------------------------------------------------------------------------
def geology_files() -> dict[int, Path]:
    cfg = json.loads(GEOLOGIES_CFG.read_text())
    return {int(g["geology_index"]): Path(g["geology_h5_file"])
            for g in cfg["geologies"]}


def load_geology(h5_path: Path):
    """(log10_perm, valid_mask) clipped to MAX_Z, shape (Z, X=70, Y=76)."""
    with h5py.File(h5_path, "r") as f:
        perm = f["Input/PermX"][:MAX_Z].astype(np.float32)
        active = f["Input/IsActive"][:MAX_Z]
    valid = (active > 0) & (perm > 0)
    logp = np.full(perm.shape, np.nan, dtype=np.float32)
    logp[valid] = np.log10(perm[valid])
    return logp, valid


def load_config_wells(it: int, sid_suffix: str):
    """[(x, y, z_depth, is_injector)] from a snapshot JSON."""
    snap_dir = RUN_ROOT / "acquire" / f"iter_{it:04d}" / "snapshots_json"
    matches = list(snap_dir.glob(f"*{sid_suffix}.json"))
    assert len(matches) == 1, f"{snap_dir} -> {sid_suffix}: {matches}"
    snap = json.loads(matches[0].read_text())
    return [(float(w["x"]), float(w["y"]), float(w["z"]),
             w["type"] == "injector") for w in snap["wells"]]


class GridMap:
    """Grid (x, y, z) -> Manim coords, Z flipped so the surface is up."""

    def __init__(self, nx: int, ny: int, nz: int):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.scale = 2.0 * MANIM_HALF_WIDTH / max(nx, ny, nz)
        self.offset = np.array([nx / 2.0, ny / 2.0, nz / 2.0])

    def to_manim(self, coords: np.ndarray) -> np.ndarray:
        m = (np.asarray(coords, dtype=np.float64) - self.offset) * self.scale
        m[..., 2] *= -1.0
        return m


# ---------------------------------------------------------------------------
# Mobject builders
# ---------------------------------------------------------------------------
def build_perm_slices(logp, valid, gm: GridMap, depths,
                      p_lo, p_hi) -> VGroup:
    # one square per active grid cell — integer subsampling (the MasterScene
    # linspace(...).astype(int) approach at resolution < grid size) leaves
    # visible gaps wherever the int step jumps by 2
    sq_size = 1.02 * gm.scale  # slight overlap hides anti-aliasing seams
    slices = VGroup()
    for zl in depths:
        layer = VGroup()
        for i in range(gm.nx):
            for j in range(gm.ny):
                if not valid[zl, i, j]:
                    continue
                t = (logp[zl, i, j] - p_lo) / (p_hi - p_lo)
                pos = gm.to_manim(np.array([[i, j, zl]]))[0]
                sq = Square(side_length=sq_size, stroke_width=0,
                            fill_color=_perm_color(t), fill_opacity=SLICE_OPACITY,
                            shade_in_3d=True)
                sq.move_to(pos)
                layer.add(sq)
        slices.add(layer)
    return slices


def build_well_lines(wells, gm: GridMap) -> VGroup:
    lines = VGroup()
    for x, y, z, is_inj in wells:
        top = gm.to_manim(np.array([[x, y, 0.0]]))[0]
        bot = gm.to_manim(np.array([[x, y, min(z, MAX_Z - 1)]]))[0]
        lines.add(Line3D(start=top, end=bot,
                         color=COLOR_INJ if is_inj else COLOR_PROD,
                         thickness=WELL_THICKNESS))
    return lines


def make_colorbar(p_lo: float, p_hi: float) -> VGroup:
    bar = Rectangle(height=3, width=0.2, stroke_color=WHITE, stroke_width=1)
    bar.set_fill([COLOR_LO, COLOR_MID, COLOR_HI], opacity=1)
    lo, hi = (round(v, 1) + 0.0 for v in (p_lo, p_hi))  # normalize -0.0 -> 0.0
    t_min = Text(f"{lo:.1f}", font_size=24).next_to(bar, DOWN, buff=0.1)
    t_max = Text(f"{hi:.1f}", font_size=24).next_to(bar, UP, buff=0.1)
    t_label = Text("log10(PermX)", font_size=20).next_to(t_max, UP, buff=0.2)
    return VGroup(bar, t_min, t_max, t_label)


# ---------------------------------------------------------------------------
class ConfigGeologyDiversity(ThreeDScene):
    def construct(self):
        geo_files = geology_files()
        geologies = [load_geology(geo_files[g]) for g in GEO_INDICES]
        configs = [load_config_wells(it, sfx) for it, sfx in CONFIGS]

        nz, nx, ny = geologies[0][0].shape  # (Z, X, Y) after clip
        gm = GridMap(nx=nx, ny=ny, nz=nz)
        depths = np.linspace(0, nz - 1, SLICE_COUNT).astype(int)

        # shared color scale across all shown geologies
        pooled = np.concatenate([lp[v] for lp, v in geologies])
        p_lo, p_hi = np.percentile(pooled, [2, 98])

        slice_groups = [build_perm_slices(lp, v, gm, depths, p_lo, p_hi)
                        for lp, v in geologies]
        well_groups = [build_well_lines(w, gm) for w in configs]

        self.set_camera_orientation(phi=75 * np.pi / 180,
                                    theta=-45 * np.pi / 180, zoom=1.0)

        colorbar = make_colorbar(p_lo, p_hi)
        self.add_fixed_in_frame_mobjects(colorbar)
        colorbar.to_edge(RIGHT, buff=0.5)

        # live mobjects that get Transform-ed in place
        slices = slice_groups[0]
        wells = well_groups[0]
        self.add(slices, wells)

        self.play(FadeIn(slices), FadeIn(wells), FadeIn(colorbar), run_time=2)
        self.begin_ambient_camera_rotation(rate=ROTATION_RATE)
        self.wait(2)

        # Phase 1: same wells, geology morphs through the ensemble
        for nxt in slice_groups[1:]:
            self.play(Transform(slices, nxt), run_time=2)
            self.wait(1.5)

        # brief hold between phases (caption space — captions added in post)
        self.wait(1.5)

        # Phase 2: same geology, well configuration morphs
        for nxt in well_groups[1:]:
            self.play(Transform(wells, nxt), run_time=1.5)
            self.wait(1.5)

        self.wait(3)
        self.stop_ambient_camera_rotation()
