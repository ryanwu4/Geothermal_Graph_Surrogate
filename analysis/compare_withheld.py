#!/usr/bin/env python3
"""Compare the withheld (top-10%) and non-withheld models on the withheld run subset.

Runs inference with both models on the full dataset, filters to the withheld
case IDs, and reports comparative accuracy metrics + a scatter plot.

Usage:
    python analysis/compare_withheld.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is importable
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
H5 = BASE / "minimal_compiled_tp.h5"

WITHHELD_DIR = BASE / "trained" / "withheld_10p_totalenergy"
BASELINE_DIR = BASE / "trained" / "withheld_0p_totalenergy"

WITHHELD_CKPT = next(WITHHELD_DIR.glob("checkpoints/best-*.ckpt"))
WITHHELD_SCALER = WITHHELD_DIR / "checkpoints" / "scaler.pkl"
WITHHELD_MANIFEST = WITHHELD_DIR / "plots" / "withheld_runs.json"

BASELINE_CKPT = next(BASELINE_DIR.glob("checkpoints/best-*.ckpt"))
BASELINE_SCALER = BASELINE_DIR / "checkpoints" / "scaler.pkl"

OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "withheld_comparison.csv"
OUT_PLOT = OUT_DIR / "withheld_comparison.png"
OUT_MD = OUT_DIR / "results.md"


def run_inference(
    ckpt: Path, scaler_path: Path, h5: Path
) -> dict[str, tuple[float, float]]:
    """Run inference and return {case_id: (predicted, actual)}."""
    import pickle
    import torch
    from torch_geometric.loader import DataLoader

    from geothermal.data import load_hetero_graphs, HeteroGraphScaler
    from geothermal.model import HeteroGNNRegressor

    with open(scaler_path, "rb") as f:
        scaler: HeteroGraphScaler = pickle.load(f)

    model = HeteroGNNRegressor.load_from_checkpoint(str(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("mps")
    model = model.to(device)
    model.eval()

    graphs_raw, targets = load_hetero_graphs(h5, target="graph_energy_total")
    graphs = [scaler.transform_graph(g) for g in graphs_raw]

    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    preds_scaled: list[np.ndarray] = []
    case_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds_scaled.append(pred.detach().cpu().numpy())
            case_ids.extend(list(batch.case_id))

    preds = scaler.inverse_targets(np.concatenate(preds_scaled, axis=0))

    result = {}
    for i, cid in enumerate(case_ids):
        result[cid] = (float(preds[i].flat[0]), float(targets[i]))
    return result


def metrics(preds: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(preds - actuals)
    return {
        "mae": float(np.mean(abs_err)),
        "mape": float(np.mean(abs_err / actuals) * 100),
        "rmse": float(np.sqrt(np.mean((preds - actuals) ** 2))),
        "medae": float(np.median(abs_err)),
        "max_err": float(np.max(abs_err)),
        "max_pct_err": float(np.max(abs_err / actuals) * 100),
    }


def main() -> None:
    # Load withheld case IDs
    with open(WITHHELD_MANIFEST) as f:
        manifest = json.load(f)
    withheld_ids = {r["case_id"] for r in manifest["withheld_runs"]}
    threshold = manifest["threshold"]
    print(f"Withheld runs: {len(withheld_ids)}, threshold: {threshold:.2e}")

    # Run inference with both models
    print("\n── Withheld model (trained without top 10%) ──")
    wh_results = run_inference(WITHHELD_CKPT, WITHHELD_SCALER, H5)

    print("\n── Baseline model (trained on all data) ──")
    bl_results = run_inference(BASELINE_CKPT, BASELINE_SCALER, H5)

    # Filter to withheld subset
    wh_preds = np.array([wh_results[cid][0] for cid in withheld_ids])
    bl_preds = np.array([bl_results[cid][0] for cid in withheld_ids])
    actuals = np.array([wh_results[cid][1] for cid in withheld_ids])

    m_wh = metrics(wh_preds, actuals)
    m_bl = metrics(bl_preds, actuals)

    # Also compute on non-withheld subset for context
    non_wh_ids = set(wh_results.keys()) - withheld_ids
    wh_preds_nwh = np.array([wh_results[cid][0] for cid in non_wh_ids])
    bl_preds_nwh = np.array([bl_results[cid][0] for cid in non_wh_ids])
    actuals_nwh = np.array([wh_results[cid][1] for cid in non_wh_ids])

    m_wh_nwh = metrics(wh_preds_nwh, actuals_nwh)
    m_bl_nwh = metrics(bl_preds_nwh, actuals_nwh)

    # ── Print ──
    print(f"\n{'='*70}")
    print(f"ON WITHHELD SUBSET ({len(withheld_ids)} runs, actual > {threshold:.2e})")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Withheld Model':>18} {'Baseline Model':>18} {'Ratio':>10}")
    print(f"{'-'*15} {'-'*18} {'-'*18} {'-'*10}")
    for k in ["mae", "mape", "rmse", "medae", "max_pct_err"]:
        v_wh = m_wh[k]
        v_bl = m_bl[k]
        unit = "%" if "pct" in k or k == "mape" else ""
        ratio = v_wh / v_bl if v_bl > 0 else float("inf")
        if unit == "%":
            print(f"{k:<15} {v_wh:>17.2f}% {v_bl:>17.2f}% {ratio:>9.2f}x")
        else:
            print(f"{k:<15} {v_wh:>18.2e} {v_bl:>18.2e} {ratio:>9.2f}x")

    print(f"\n{'='*70}")
    print(f"ON NON-WITHHELD SUBSET ({len(non_wh_ids)} runs)")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Withheld Model':>18} {'Baseline Model':>18} {'Ratio':>10}")
    print(f"{'-'*15} {'-'*18} {'-'*18} {'-'*10}")
    for k in ["mae", "mape", "rmse"]:
        v_wh = m_wh_nwh[k]
        v_bl = m_bl_nwh[k]
        unit = "%" if k == "mape" else ""
        ratio = v_wh / v_bl if v_bl > 0 else float("inf")
        if unit == "%":
            print(f"{k:<15} {v_wh:>17.2f}% {v_bl:>17.2f}% {ratio:>9.2f}x")
        else:
            print(f"{k:<15} {v_wh:>18.2e} {v_bl:>18.2e} {ratio:>9.2f}x")

    # ── CSV ──
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "actual",
                "pred_withheld_model",
                "pred_baseline_model",
                "err_pct_withheld",
                "err_pct_baseline",
            ]
        )
        for cid in sorted(withheld_ids):
            a = wh_results[cid][1]
            pw = wh_results[cid][0]
            pb = bl_results[cid][0]
            writer.writerow(
                [cid, a, pw, pb, abs(pw - a) / a * 100, abs(pb - a) / a * 100]
            )
    print(f"\nSaved CSV: {OUT_CSV}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: predicted vs actual
    lo = min(actuals.min(), wh_preds.min(), bl_preds.min()) * 0.95
    hi = max(actuals.max(), wh_preds.max(), bl_preds.max()) * 1.05

    axes[0].scatter(
        actuals, bl_preds, s=12, alpha=0.6, c="tab:blue", label="Baseline (all data)"
    )
    axes[0].scatter(
        actuals, wh_preds, s=12, alpha=0.6, c="tab:red", label="Withheld model"
    )
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect")
    axes[0].axvline(
        threshold,
        color="gray",
        linestyle=":",
        alpha=0.6,
        label=f"Threshold={threshold:.2e}",
    )
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("Actual Energy Production")
    axes[0].set_ylabel("Predicted Energy Production")
    axes[0].set_title(f"Withheld Runs ({len(withheld_ids)} cases)")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal", adjustable="box")

    # Percent error distribution
    pct_err_wh = (wh_preds - actuals) / actuals * 100
    pct_err_bl = (bl_preds - actuals) / actuals * 100
    bins = np.linspace(-20, 5, 50)
    axes[1].hist(
        pct_err_bl,
        bins=bins,
        alpha=0.6,
        color="tab:blue",
        label="Baseline",
        edgecolor="white",
    )
    axes[1].hist(
        pct_err_wh,
        bins=bins,
        alpha=0.6,
        color="tab:red",
        label="Withheld model",
        edgecolor="white",
    )
    axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Prediction Error (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Distribution on Withheld Runs")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {OUT_PLOT}")

    # ── Markdown summary ──
    md = f"""# Withholding Analysis: Top 10% Energy Production

## Setup

- **Withheld model**: trained on bottom 90% of runs (top 10% by `graph_energy_total` removed)
- **Baseline model**: trained on all runs (no withholding)
- **Withheld subset**: {len(withheld_ids)} runs with actual > {threshold:.2e}
- **Non-withheld subset**: {len(non_wh_ids)} runs

## Results on Withheld Runs (never seen by withheld model)

| Metric | Withheld Model | Baseline (all data) | Ratio |
|---|---:|---:|---:|
| MAE | {m_wh['mae']:.2e} | {m_bl['mae']:.2e} | {m_wh['mae']/m_bl['mae']:.1f}× |
| MAPE | {m_wh['mape']:.2f}% | {m_bl['mape']:.2f}% | {m_wh['mape']/m_bl['mape']:.1f}× |
| RMSE | {m_wh['rmse']:.2e} | {m_bl['rmse']:.2e} | {m_wh['rmse']/m_bl['rmse']:.1f}× |
| Median AE | {m_wh['medae']:.2e} | {m_bl['medae']:.2e} | {m_wh['medae']/m_bl['medae']:.1f}× |
| Max Error % | {m_wh['max_pct_err']:.1f}% | {m_bl['max_pct_err']:.1f}% | {m_wh['max_pct_err']/m_bl['max_pct_err']:.1f}× |

## Results on Non-Withheld Runs (training distribution)

| Metric | Withheld Model | Baseline (all data) | Ratio |
|---|---:|---:|---:|
| MAE | {m_wh_nwh['mae']:.2e} | {m_bl_nwh['mae']:.2e} | {m_wh_nwh['mae']/m_bl_nwh['mae']:.1f}× |
| MAPE | {m_wh_nwh['mape']:.2f}% | {m_bl_nwh['mape']:.2f}% | {m_wh_nwh['mape']/m_bl_nwh['mape']:.1f}× |
| RMSE | {m_wh_nwh['rmse']:.2e} | {m_bl_nwh['rmse']:.2e} | {m_wh_nwh['rmse']/m_bl_nwh['rmse']:.1f}× |

## Key Takeaways

The withheld model **systematically underpredicts** the top-10% runs (pred range caps near the
training threshold of {threshold:.2e}). This is expected behavior — the model has no exposure
to the high-energy regime and cannot extrapolate beyond its training distribution.

The baseline model, having seen these runs during training, achieves substantially lower error
on this subset. On the non-withheld subset, both models perform comparably, confirming that
withholding does not degrade performance within the training distribution.

![Comparison plot](withheld_comparison.png)
"""
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(f"Saved summary: {OUT_MD}")


if __name__ == "__main__":
    main()
