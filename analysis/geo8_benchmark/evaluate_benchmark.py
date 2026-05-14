"""Run the trained surrogate over the benchmark.h5 cases and produce per-cohort metrics + plots.

Outputs (in analysis/geo8_benchmark/eval/):
  metrics.json                     — aggregate / geo8 / non-geo8 metric tables
  predictions.csv                  — per-case predicted/actual/abs_err/pct_err + geo_idx + split
  scatter_all_by_cohort.png        — pred-vs-actual, geo8 vs non-geo8 colored
  scatter_geo8_only.png            — pred-vs-actual for geo 8 cases only
  error_dist_abs.png               — histograms of abs error, geo8 vs non-geo8
  error_dist_pct.png               — histograms of pct error, geo8 vs non-geo8
  residuals_kde.png                — KDE of residuals, geo8 vs non-geo8
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

import sys
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))  # surrogate repo root
from geothermal.data import HeteroGraphScaler, load_hetero_graphs, split_indices_stratified, apply_ablation
from geothermal.model import HeteroGNNRegressor

BENCHMARK_H5 = THIS / "benchmark.h5"
MANIFEST = THIS / "benchmark_manifest.json"
TARGET = "graph_discounted_net_revenue"


def _mape(y_true, y_pred, threshold_frac=0.01):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    mask = np.abs(yt) > (np.abs(np.mean(yt)) * threshold_frac)
    if not np.any(mask):
        return float("nan")
    raw = np.abs((yt[mask] - yp[mask]) / np.maximum(np.abs(yt[mask]), 1e-12)) * 100.0
    clipped = np.clip(raw, a_min=None, a_max=np.percentile(raw, 99))
    return float(np.mean(clipped))


def _metrics(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return {
        "n": int(yt.size),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mape": _mape(yt, yp),
        "r2": float(r2_score(yt, yp)) if yt.size >= 2 else float("nan"),
        "mean_true": float(np.mean(yt)),
        "mean_pred": float(np.mean(yp)),
    }


def find_ckpt_and_scaler(training_root: Path) -> tuple[Path, Path]:
    ckpt_dir = training_root / "geothermal_hetero_gnn" / "run_00" / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No best-*.ckpt in {ckpt_dir}")
    ckpt = ckpts[0]
    scaler_path = ckpt.parent / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"No scaler.pkl in {ckpt.parent}")
    return ckpt, scaler_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path,
                    default=THIS / "runs" / "training",
                    help="Root used as train.py --output-root")
    ap.add_argument("--out-dir", type=Path, default=THIS / "runs" / "eval")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--node-encoder", choices=["profile", "cnn", "hybrid"], default="cnn",
                    help="Must match the value used at training time.")
    ap.add_argument("--ablate", nargs="+", default=[],
                    help="Ablation group names to zero out (same flag values as train.py).")
    ap.add_argument("--enrich-global-attr",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Must match the flag used at training time.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graphs from {BENCHMARK_H5} (node_encoder={args.node_encoder}, ablate={args.ablate}, enrich={args.enrich_global_attr}) ...")
    graphs, targets = load_hetero_graphs(
        BENCHMARK_H5, target=TARGET, node_encoder=args.node_encoder,
        enrich_global_attr=args.enrich_global_attr,
    )
    if args.ablate:
        apply_ablation(graphs, args.ablate)
    print(f"  {len(graphs)} graphs loaded, target range [{targets.min():.3e}, {targets.max():.3e}]")

    # Reconstruct the same stratified split train.py used
    train_idx, val_idx, test_idx = split_indices_stratified(
        targets=targets,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    split_of = {}
    for i in train_idx: split_of[i] = "train"
    for i in val_idx:   split_of[i] = "val"
    for i in test_idx:  split_of[i] = "test"

    # Load checkpoint + scaler
    ckpt_path, scaler_path = find_ckpt_and_scaler(args.training_root)
    print(f"Loading checkpoint {ckpt_path}")
    print(f"Loading scaler     {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler: HeteroGraphScaler = pickle.load(f)

    model = HeteroGNNRegressor.load_from_checkpoint(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Transform all graphs with the loaded scaler
    transformed = [scaler.transform_graph(g) for g in graphs]
    loader = DataLoader(transformed, batch_size=16, shuffle=False)

    preds_scaled = []
    trues_scaled = []
    case_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            p = model(batch).detach().cpu().numpy()
            preds_scaled.append(p)
            trues_scaled.append(batch.y.detach().cpu().numpy())
            case_ids.extend(list(batch.case_id))
    preds_scaled = np.concatenate(preds_scaled)
    trues_scaled = np.concatenate(trues_scaled)

    y_pred = scaler.inverse_targets(preds_scaled).ravel()
    y_true = scaler.inverse_targets(trues_scaled).ravel()

    manifest = json.loads(MANIFEST.read_text())
    geo_idx = np.array([manifest[c] for c in case_ids], dtype=int)
    is_geo8 = (geo_idx == 8)
    split_lbl = np.array([split_of.get(i, "?") for i in range(len(graphs))])
    # Map split labels by case order (graphs are in load order, matching case_ids)
    split_lbl = split_lbl[:len(case_ids)]

    abs_err = np.abs(y_pred - y_true)
    pct_err = (y_pred - y_true) / np.maximum(np.abs(y_true), 1e-12) * 100.0

    # Save per-case predictions CSV
    import csv as _csv
    with open(args.out_dir / "predictions.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["case_id", "geology_index", "split", "y_true", "y_pred", "abs_err", "pct_err"])
        for cid, g, s, yt, yp, ae, pe in zip(case_ids, geo_idx, split_lbl, y_true, y_pred, abs_err, pct_err):
            w.writerow([cid, int(g), s, float(yt), float(yp), float(ae), float(pe)])

    # Metrics
    metrics = {
        "aggregate": _metrics(y_true, y_pred),
        "geo8": _metrics(y_true[is_geo8], y_pred[is_geo8]),
        "non_geo8": _metrics(y_true[~is_geo8], y_pred[~is_geo8]),
        "by_split": {
            split: _metrics(y_true[split_lbl == split], y_pred[split_lbl == split])
            for split in ("train", "val", "test")
        },
        "geo8_by_split": {
            split: _metrics(y_true[is_geo8 & (split_lbl == split)],
                            y_pred[is_geo8 & (split_lbl == split)])
            for split in ("train", "val", "test")
        },
        "non_geo8_by_split": {
            split: _metrics(y_true[(~is_geo8) & (split_lbl == split)],
                            y_pred[(~is_geo8) & (split_lbl == split)])
            for split in ("train", "val", "test")
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    # ---- Plots ----
    MUSD = 1e6  # convert from $ -> M$

    def _fmt(ax, title):
        ax.set_xlabel("Actual revenue (M$)")
        ax.set_ylabel("Predicted revenue (M$)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    # Combined scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true[~is_geo8] / MUSD, y_pred[~is_geo8] / MUSD,
               s=18, alpha=0.45, color="tab:blue",
               label=f"non-geo8 (n={int(np.sum(~is_geo8))}, MAPE={metrics['non_geo8']['mape']:.1f}%)")
    ax.scatter(y_true[is_geo8] / MUSD, y_pred[is_geo8] / MUSD,
               s=22, alpha=0.7, color="tab:red",
               label=f"geo 8 (n={int(np.sum(is_geo8))}, MAPE={metrics['geo8']['mape']:.1f}%)")
    lo = min(y_true.min(), y_pred.min()) / MUSD
    hi = max(y_true.max(), y_pred.max()) / MUSD
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    _fmt(ax, "Predicted vs actual — geo 8 vs cohort")
    fig.tight_layout()
    fig.savefig(args.out_dir / "scatter_all_by_cohort.png", dpi=160)
    plt.close(fig)

    # Geo 8 only scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    # Color by split for geo 8
    split_colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for s in ("train", "val", "test"):
        m = is_geo8 & (split_lbl == s)
        if np.any(m):
            ax.scatter(y_true[m] / MUSD, y_pred[m] / MUSD,
                       s=24, alpha=0.7, color=split_colors[s],
                       label=f"{s} (n={int(np.sum(m))})")
    y_g8 = y_true[is_geo8] / MUSD
    p_g8 = y_pred[is_geo8] / MUSD
    lo8 = min(y_g8.min(), p_g8.min())
    hi8 = max(y_g8.max(), p_g8.max())
    ax.plot([lo8, hi8], [lo8, hi8], "k--", lw=1, label="y = x")
    _fmt(ax, f"Geo 8 only — MAPE={metrics['geo8']['mape']:.1f}%, MAE={metrics['geo8']['mae']/MUSD:.1f} M$")
    fig.tight_layout()
    fig.savefig(args.out_dir / "scatter_geo8_only.png", dpi=160)
    plt.close(fig)

    # Absolute error histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, np.percentile(abs_err / MUSD, 99), 40)
    ax.hist(abs_err[~is_geo8] / MUSD, bins=bins, alpha=0.55, color="tab:blue",
            label=f"non-geo8 (mean={metrics['non_geo8']['mae']/MUSD:.1f} M$)", density=True)
    ax.hist(abs_err[is_geo8] / MUSD, bins=bins, alpha=0.55, color="tab:red",
            label=f"geo 8 (mean={metrics['geo8']['mae']/MUSD:.1f} M$)", density=True)
    ax.set_xlabel("|prediction - actual| (M$)")
    ax.set_ylabel("Density")
    ax.set_title("Absolute error distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "error_dist_abs.png", dpi=160)
    plt.close(fig)

    # Percent error histogram (clipped to ±200)
    fig, ax = plt.subplots(figsize=(9, 5))
    pe_clip = np.clip(pct_err, -200, 200)
    bins = np.linspace(-200, 200, 60)
    ax.hist(pe_clip[~is_geo8], bins=bins, alpha=0.55, color="tab:blue",
            label=f"non-geo8 (MAPE={metrics['non_geo8']['mape']:.1f}%)", density=True)
    ax.hist(pe_clip[is_geo8], bins=bins, alpha=0.55, color="tab:red",
            label=f"geo 8 (MAPE={metrics['geo8']['mape']:.1f}%)", density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Percentage error (pred - actual) / actual (%, clipped ±200)")
    ax.set_ylabel("Density")
    ax.set_title("Percentage error distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "error_dist_pct.png", dpi=160)
    plt.close(fig)

    # Residual KDE
    from scipy.stats import gaussian_kde
    res = (y_pred - y_true) / MUSD
    fig, ax = plt.subplots(figsize=(9, 5))
    for mask, color, label in [
        (~is_geo8, "tab:blue", f"non-geo8 (bias={np.mean(res[~is_geo8]):+.1f} M$)"),
        (is_geo8, "tab:red", f"geo 8 (bias={np.mean(res[is_geo8]):+.1f} M$)"),
    ]:
        if np.sum(mask) >= 2:
            kde = gaussian_kde(res[mask])
            xs = np.linspace(np.percentile(res, 1), np.percentile(res, 99), 400)
            ax.plot(xs, kde(xs), color=color, lw=2, label=label)
            ax.fill_between(xs, 0, kde(xs), color=color, alpha=0.18)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Residual = pred - actual (M$)")
    ax.set_ylabel("Density")
    ax.set_title("Residual KDE — overprediction shows positive bias")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "residuals_kde.png", dpi=160)
    plt.close(fig)

    # Per-geology MAE bar
    fig, ax = plt.subplots(figsize=(10, 5))
    geos = sorted(np.unique(geo_idx))
    maes = [np.mean(abs_err[geo_idx == g]) / MUSD for g in geos]
    colors = ["tab:red" if g == 8 else "tab:blue" for g in geos]
    ax.bar(geos, maes, color=colors)
    ax.set_xticks(geos)
    ax.set_xlabel("geology index")
    ax.set_ylabel("MAE (M$)")
    ax.set_title("Per-geology MAE")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "per_geology_mae.png", dpi=160)
    plt.close(fig)

    print(f"All artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
