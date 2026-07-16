"""Stage-1 ablation comparison figures (reference palette, light mode)."""
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate/analysis/surrogate_ablation")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
VARIANTS = ["gnn_default", "gnn_dist_edge", "gnn_type_only_nodes",
            "gnn_svd_edge", "mlp", "global_cnn"]
V_COLORS = dict(zip(VARIANTS,
    ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]))
V_LABELS = {
    "gnn_default": "GNN default (CNN edges+nodes)",
    "gnn_dist_edge": "GNN, distance-only edges",
    "gnn_type_only_nodes": "GNN, type-only nodes",
    "gnn_svd_edge": "GNN, SVD edges",
    "mlp": "MLP (positions + geology ID)",
    "global_cnn": "Global 3D CNN",
}
DATASETS = ["simple_2pair", "hard_fulldepth", "highperf_cma"]
D_LABELS = {"simple_2pair": "SIMPLE", "hard_fulldepth": "HARD", "highperf_cma": "HIGH-PERF"}
TARGETS = ["graph_energy_total", "graph_discounted_net_revenue", "node_wept_final"]
T_LABELS = {"graph_energy_total": "Total energy production",
            "graph_discounted_net_revenue": "Discounted revenue",
            "node_wept_final": "Per-producer WEPT (year 30)"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": SECONDARY, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "sans-serif", "legend.frameon": False,
})

rows = list(csv.DictReader(open(ROOT / "runs/stage1/extended_metrics.csv")))
def get(ds, tgt, var, key):
    r = next(x for x in rows if x["dataset"] == ds and x["target"] == tgt
             and x["variant"] == var)
    v = r[key]
    return float(v) if v != "" else np.nan


def grouped_bars(ax, tgt, key, ylabel=None):
    ng, nv = len(DATASETS), len(VARIANTS)
    width = 0.8 / nv
    for j, var in enumerate(VARIANTS):
        vals = [get(ds, tgt, var, key) for ds in DATASETS]
        x = np.arange(ng) + (j - (nv - 1) / 2) * width
        ax.bar(x, vals, width * 0.9, color=V_COLORS[var], zorder=3)
    ax.set_xticks(np.arange(ng))
    ax.set_xticklabels([D_LABELS[d] for d in DATASETS])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


# ---------------- Figure 1: accuracy comparison ----------------
fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.0))
fig.subplots_adjust(hspace=0.38, wspace=0.24, bottom=0.14)
for c, tgt in enumerate(TARGETS):
    ax = axes[0, c]
    grouped_bars(ax, tgt, "wmape", ylabel="test wMAPE (%)" if c == 0 else None)
    ax.set_title(T_LABELS[tgt], color=INK, loc="left")
    ax = axes[1, c]
    grouped_bars(ax, tgt, "r2", ylabel="test R²" if c == 0 else None)
    ax.set_ylim(0, 1.0)
handles = [plt.Rectangle((0, 0), 1, 1, color=V_COLORS[v], label=V_LABELS[v])
           for v in VARIANTS]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Architecture ablation — test accuracy by dataset and objective "
             "(seed 42, identical splits)", color=INK, fontsize=13,
             fontweight="bold", y=0.99)
fig.savefig(FIG / "ablation_accuracy.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: ranking quality (revenue objective) ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
fig.subplots_adjust(wspace=0.24, bottom=0.3)
tgt = "graph_discounted_net_revenue"
grouped_bars(axes[0], tgt, "spearman", ylabel="Spearman ρ (test)")
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title("Rank correlation — discounted revenue", color=INK, loc="left")
grouped_bars(axes[1], tgt, "top10_recall", ylabel="top-10% recall (test)")
axes[1].set_ylim(0, 1.0)
axes[1].set_title("Top-decile identification — discounted revenue", color=INK, loc="left")
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           labelcolor=SECONDARY, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Architecture ablation — optimization-relevant ranking quality",
             color=INK, fontsize=13, fontweight="bold", y=1.02)
fig.savefig(FIG / "ablation_ranking.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("wrote", FIG / "ablation_accuracy.png")
print("wrote", FIG / "ablation_ranking.png")
