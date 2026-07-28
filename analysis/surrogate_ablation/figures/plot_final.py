"""Final ablation figures: architecture comparison (7 variants incl. no-CNN),
data-efficiency curves, and held-out-geology OOD generalization."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate/analysis/surrogate_ablation")
FIG = ROOT / "figures"

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
VARIANTS = ["gnn_default", "gnn_type_only_nodes", "gnn_dist_edge", "gnn_svd_edge",
            "gnn_no_cnn", "global_cnn", "mlp"]
V_COLORS = {
    "gnn_default": "#2a78d6", "gnn_type_only_nodes": "#1baf7a",
    "gnn_dist_edge": "#6da7ec", "gnn_svd_edge": "#008300",
    "gnn_no_cnn": "#eb6834", "global_cnn": "#e34948", "mlp": "#4a3aa7",
}
V_LABELS = {
    "gnn_default": "GNN default (CNN edges+nodes)",
    "gnn_type_only_nodes": "GNN, type-only nodes",
    "gnn_dist_edge": "GNN, distance-only edges",
    "gnn_svd_edge": "GNN, SVD edges",
    "gnn_no_cnn": "GNN, no CNN (dist edges + type nodes)",
    "global_cnn": "Global 3D CNN",
    "mlp": "MLP (positions + geology ID)",
}
DATASETS = ["simple_2pair", "hard_fulldepth", "highperf_cma"]
D_LABELS = {"simple_2pair": "SIMPLE", "hard_fulldepth": "HARD", "highperf_cma": "HIGH-PERF"}
T_LABELS = {"graph_energy_total": "Total energy",
            "graph_discounted_net_revenue": "Discounted revenue",
            "node_wept_final": "Per-producer WEPT"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": SECONDARY, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "axes.titlesize": 10.5, "axes.labelsize": 10, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "sans-serif", "legend.frameon": False,
})


def load(stage):
    return list(csv.DictReader(open(ROOT / "runs" / stage / "summary.csv")))


def agg(rows, split, dataset, target, variant, size, hgeo, field="r2"):
    """mean, std over seeds."""
    vals = [float(r[field]) for r in rows
            if r["split"] == split and r["dataset"] == dataset
            and r["target"] == target and r["variant"] == variant
            and r["train_size"] == size and r["holdout_geo"] == hgeo]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


s1 = load("stage1")
s2 = load("stage2_data_efficiency")
s3 = load("stage3_holdout_geo")
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=V_COLORS[v], label=V_LABELS[v])
                  for v in VARIANTS]

# ================= Figure 1: architecture comparison (full data, test R2) =================
TARGETS = ["graph_energy_total", "graph_discounted_net_revenue", "node_wept_final"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
fig.subplots_adjust(wspace=0.16, bottom=0.30, top=0.86)
for c, tgt in enumerate(TARGETS):
    ax = axes[c]
    nv = len(VARIANTS)
    width = 0.8 / nv
    for j, v in enumerate(VARIANTS):
        vals, errs = [], []
        for d in DATASETS:
            m, sd = agg(s1, "test", d, tgt, v, "full", "")
            vals.append(m if m is not None else np.nan)
            errs.append(sd if sd else 0)
        x = np.arange(len(DATASETS)) + (j - (nv - 1) / 2) * width
        ax.bar(x, vals, width * 0.9, yerr=errs, color=V_COLORS[v], zorder=3,
               error_kw={"lw": 0.7, "ecolor": MUTED})
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([D_LABELS[d] for d in DATASETS])
    ax.set_title(T_LABELS[tgt], color=INK, loc="left")
    ax.set_ylim(0, 1.0)
    if c == 0:
        ax.set_ylabel("test R²  (mean ± sd over seeds)")
    ax.axhline(0, color=BASELINE, lw=0.8)
    style(ax)
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5,
           labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Architecture ablation — full-data test accuracy (R²) across datasets and objectives",
             color=INK, fontsize=13, fontweight="bold", y=0.98)
fig.savefig(FIG / "final_architecture_r2.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ================= Figure 2: data-efficiency curves =================
# x = {256, 512, full}; y = test R2; targets revenue+wept; 3 datasets.
DE_TARGETS = ["graph_discounted_net_revenue", "node_wept_final"]
sizes = ["256", "512", "full"]
xpos = [0, 1, 2]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
fig.subplots_adjust(hspace=0.28, wspace=0.16, bottom=0.16, top=0.9)
for r, tgt in enumerate(DE_TARGETS):
    for c, d in enumerate(DATASETS):
        ax = axes[r, c]
        for v in VARIANTS:
            ys, es = [], []
            for size in sizes:
                src = s1 if size == "full" else s2
                m, sd = agg(src, "test", d, tgt, v, size, "")
                ys.append(m if m is not None else np.nan)
                es.append(sd if sd else 0)
            ax.errorbar(xpos, ys, yerr=es, color=V_COLORS[v], lw=1.8, marker="o",
                        ms=4, capsize=2, elinewidth=0.7, zorder=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["256", "512", "full"])
        if r == 1:
            ax.set_xlabel("train-set size")
        if c == 0:
            ax.set_ylabel(f"{T_LABELS[tgt]}\ntest R²")
        if r == 0:
            ax.set_title(D_LABELS[d], color=INK)
        ax.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5,
           labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Data efficiency — test R² vs train-set size (256 / 512 / full)",
             color=INK, fontsize=13, fontweight="bold", y=0.965)
fig.savefig(FIG / "final_data_efficiency.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ================= Figure 3: held-out-geology OOD =================
# rows: revenue, wept ; cols: datasets ; per variant: in-dist test vs OOD (geo8 fold)
# plus a geo3 (typical) reference via lighter marker. Show test_ood R2 for both folds.
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
fig.subplots_adjust(hspace=0.33, wspace=0.16, bottom=0.16, top=0.9)
folds = [("3", "geo 3 (typical)"), ("8", "geo 8 (tight, hard)")]
for r, tgt in enumerate(DE_TARGETS):
    for c, d in enumerate(DATASETS):
        ax = axes[r, c]
        nv = len(VARIANTS)
        width = 0.8 / nv
        for j, v in enumerate(VARIANTS):
            vals = []
            for fold, _ in folds:
                m, _ = agg(s3, "test_ood", d, tgt, v, "full", fold)
                vals.append(m if m is not None else np.nan)
            x = np.arange(2) + (j - (nv - 1) / 2) * width
            ax.bar(x, vals, width * 0.9, color=V_COLORS[v], zorder=3)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels([f[1] for f in folds])
        ax.axhline(0, color=BASELINE, lw=0.8)
        if c == 0:
            ax.set_ylabel(f"{T_LABELS[tgt]}\nOOD test R²")
        if r == 0:
            ax.set_title(D_LABELS[d], color=INK)
        ax.set_ylim(min(-1.2, ax.get_ylim()[0]), 1.0)
        style(ax)
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8.5,
           labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Held-out-geology generalization — R² on the unseen geology "
             "(negative = worse than predicting the mean)",
             color=INK, fontsize=13, fontweight="bold", y=0.965)
fig.savefig(FIG / "final_holdout_geo.png", dpi=170, bbox_inches="tight")
plt.close(fig)

print("wrote final_architecture_r2.png, final_data_efficiency.png, final_holdout_geo.png")
