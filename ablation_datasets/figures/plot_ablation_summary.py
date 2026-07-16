"""Summary figures for the three ablation datasets: well-placement coverage
and objective-value span. Styled per the dataviz reference palette (light mode).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ABL = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate/ablation_datasets")
FIG = ABL / "figures"
d = np.load(FIG / "dataset_summary_arrays.npz")

# --- reference palette (light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"simple": "#2a78d6", "hard": "#1baf7a", "highperf": "#eda100"}
LABELS = {
    "simple": "SIMPLE — 2 pairs, LHS, depth 50–55",
    "hard": "HARD — 6 pairs, LHS, depth 11–70",
    "highperf": "HIGH-PERF — 6 pairs, CMA-optimized",
}
SHORT = {"simple": "SIMPLE", "hard": "HARD", "highperf": "HIGH-PERF"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DATASETS = ["simple", "hard", "highperf"]
FACILITIES = [(20, 30), (40, 40)]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": SECONDARY, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "sans-serif", "legend.frameon": False,
})


def style_axes(ax, grid_axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def step_density(ax, values, name, bins, logx=False):
    v = np.log10(values) if logx else values
    hist, edges = np.histogram(v, bins=bins, density=True)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(hist, 2)
    ax.plot(x, y, color=SERIES[name], lw=1.8, solid_joinstyle="miter", zorder=3)
    ax.fill_between(x, 0, y, color=SERIES[name], alpha=0.10, lw=0, zorder=2)
    return edges, hist


# ================= Figure 1: well placement coverage =================
fig = plt.figure(figsize=(12.5, 8.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.28)

# Row 1 — xy density maps (share of dataset wells per 2x2-cell bin).
# Well 'x' is the 70-cell grid axis, 'y' the 76-cell axis (sampler frame:
# nx=valid_mask.shape[1]=70, ny=shape[2]=76) → buffer x∈[10,59], y∈[10,65].
xbins = np.arange(0, 72, 2)
ybins = np.arange(0, 78, 2)
maps = {}
for name in DATASETS:
    H, _, _ = np.histogram2d(d[f"{name}_x"], d[f"{name}_y"], bins=[xbins, ybins])
    H = 100.0 * H / H.sum()
    maps[name] = H
# Shared scale capped well below HIGH-PERF's edge-pinning hotspots so the
# near-uniform LHS texture stays visible; colorbar gets an over-arrow.
vmax = 4.0 * 100.0 / ((59 - 10) / 2 * (65 - 10) / 2)  # 4x uniform share/bin

for i, name in enumerate(DATASETS):
    ax = fig.add_subplot(gs[0, i])
    im = ax.pcolormesh(xbins, ybins, maps[name].T, cmap=SEQ_BLUE, vmin=0, vmax=vmax,
                       rasterized=True)
    ax.add_patch(Rectangle((10, 10), 49, 55, fill=False, ls=(0, (4, 3)),
                           ec=MUTED, lw=0.9, zorder=4))
    for fx, fy in FACILITIES:
        ax.plot(fx, fy, marker="x", ms=7, mew=1.8, color=INK, zorder=5)
    n_wells = len(d[f"{name}_x"])
    n_cases = len(d[f"{name}_energy"])
    ax.set_title(f"{SHORT[name]}\n{n_cases:,} cases · {n_wells:,} wells", color=INK)
    ax.set_xlabel("grid x")
    if i == 0:
        ax.set_ylabel("grid y")
    ax.set_xlim(0, 70); ax.set_ylim(0, 76)
    ax.set_aspect("equal")
    style_axes(ax, grid_axis=None)
cbar = fig.colorbar(im, ax=[fig.axes[k] for k in range(3)], shrink=0.75,
                    pad=0.015, aspect=28, extend="max")
cbar.set_label("share of dataset's wells per 2×2 cell (%)", color=SECONDARY, fontsize=9)
cbar.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
cbar.outline.set_edgecolor(BASELINE)

# Row 2a — depth distributions
axd = fig.add_subplot(gs[1, 0:2])
for name in DATASETS:
    step_density(axd, d[f"{name}_depth"], name, bins=np.arange(0.5, 63.5, 1.0))
axd.set_xlabel("realized well depth (grid layer k)")
axd.set_ylabel("density")
axd.set_xlim(0, 63)
style_axes(axd)
axd.annotate("SIMPLE", xy=(52.5, axd.get_ylim()[1] * 0.92), color=SERIES["simple"],
             fontsize=9, fontweight="bold", ha="right")
axd.annotate("HIGH-PERF", xy=(45.5, axd.get_ylim()[1] * 0.55), color=SERIES["highperf"],
             fontsize=9, fontweight="bold", ha="right")
axd.annotate("HARD", xy=(25, axd.get_ylim()[1] * 0.16), color=SERIES["hard"],
             fontsize=9, fontweight="bold", ha="center")
axd.set_title("Realized well depths (perforation bottom; rock-capped)", color=INK,
              loc="left")

# Row 2b — cases per geology
axg = fig.add_subplot(gs[1, 2])
width = 0.27
geos = np.arange(15)
for j, name in enumerate(DATASETS):
    counts = np.bincount(d[f"{name}_geo"], minlength=15)
    axg.bar(geos + (j - 1) * width, counts, width * 0.92, color=SERIES[name],
            zorder=3, label=SHORT[name])
axg.set_xlabel("geology index")
axg.set_ylabel("cases")
axg.set_xticks(geos[::2])
style_axes(axg, grid_axis="y")
axg.set_title("Cases per geology", color=INK, loc="left")
axg.legend(fontsize=8, labelcolor=SECONDARY, handlelength=1.0, borderaxespad=0.0)

fig.suptitle("Ablation datasets — well-placement coverage", color=INK,
             fontsize=13, fontweight="bold", y=0.99)
fig.savefig(FIG / "well_placement_coverage.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ================= Figure 2: objective coverage =================
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
fig.subplots_adjust(wspace=0.28)

# (a) field energy production total — log10
ax = axes[0]
allv = np.concatenate([d[f"{n}_energy"] for n in DATASETS])
bins = np.linspace(np.log10(allv.min()) - 0.05, np.log10(allv.max()) + 0.05, 44)
for name in DATASETS:
    step_density(ax, d[f"{name}_energy"], name, bins, logx=True)
ax.set_xlabel("field energy production total, 30 yr (log$_{10}$ kJ)")
ax.set_ylabel("density")
ax.set_title("Total energy production", color=INK, loc="left")
style_axes(ax)

# (b) discounted net revenue — linear, M EUR
ax = axes[1]
allv = np.concatenate([d[f"{n}_revenue"] for n in DATASETS]) / 1e6
bins = np.linspace(allv.min() - 5, allv.max() + 5, 44)
for name in DATASETS:
    step_density(ax, d[f"{name}_revenue"] / 1e6, name, bins)
ax.set_xlabel("discounted net revenue (M€, 30 yr)")
ax.set_title("Discounted revenue", color=INK, loc="left")
style_axes(ax)

# (c) per-producer final-year WEPT — log10
ax = axes[2]
allv = np.concatenate([d[f"{n}_wept_final"] for n in DATASETS])
allv = allv[allv > 0]
bins = np.linspace(np.log10(allv.min()) - 0.05, np.log10(allv.max()) + 0.05, 44)
for name in DATASETS:
    v = d[f"{name}_wept_final"]
    step_density(ax, v[v > 0], name, bins, logx=True)
ax.set_xlabel("per-producer WEPT, year 30 (log$_{10}$ kJ)")
ax.set_title("Final-year well energy (producers)", color=INK, loc="left")
style_axes(ax)

handles = [plt.Line2D([], [], color=SERIES[n], lw=2.2, label=LABELS[n])
           for n in DATASETS]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
           ncol=3, fontsize=9, labelcolor=SECONDARY)
fig.suptitle("Ablation datasets — objective-value span", color=INK,
             fontsize=13, fontweight="bold", y=1.04)
fig.savefig(FIG / "objective_coverage.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("wrote", FIG / "well_placement_coverage.png")
print("wrote", FIG / "objective_coverage.png")
