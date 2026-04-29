#!/usr/bin/env python3
"""Compare surrogate CMA-ES predictions against Intersect outputs.

Outputs:
1) Scatter plots for:
   - surrogate predicted total energy vs Intersect total energy
   - surrogate predicted discounted revenue vs Intersect discounted revenue
2) Best/worst fit well-location overlays (over normalized log PermX background)
   for each geology and each metric.
3) Matched rows CSV for downstream analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from geothermal.economics import (
    discounted_revenue_from_rates,
    resolve_real_discount_rate_from_attrs,
)

CASE_RE = re.compile(r"_(?P<geo>\d+)_run(?P<run>\d+)(?:_iter(?P<iter>\d+))?$")

MANIM_BG = "#000000"
MANIM_BLUE = "#58C4DD"
MANIM_ORANGE = "#FF9000"
MANIM_WHITE = "#FFFFFF"
MANIM_GREY = "#888888"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare surrogate-estimated discounted revenue / total energy against "
            "Intersect results in intersect_preprocessed.h5"
        )
    )
    parser.add_argument(
        "--intersect-h5",
        type=Path,
        default=Path("intersect_preprocessed.h5"),
        help="Path to preprocessed Intersect output H5",
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=None,
        help=(
            "Path to CMA-ES snapshot manifest JSON. "
            "If omitted, newest snapshot_manifest_*.json under inference/inference_outputs is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/surrogate_vs_intersect"),
        help="Directory for plots and matched CSV",
    )
    return parser.parse_args()


def find_latest_manifest(repo_root: Path) -> Path:
    manifests = sorted(
        (repo_root / "inference" / "inference_outputs").glob(
            "**/snapshot_manifest_*.json"
        ),
        key=lambda p: p.stat().st_mtime,
    )
    if not manifests:
        raise FileNotFoundError("No snapshot_manifest_*.json found under inference/inference_outputs")
    return manifests[-1]


def resolve_path(raw_path: str, manifest_path: Path, repo_root: Path) -> Path:
    p = Path(raw_path)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((manifest_path.parent / p).resolve())
        # In manifests, artifact paths are usually relative to repo/inference
        candidates.append((repo_root / "inference" / p).resolve())
        candidates.append((repo_root / p).resolve())

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve path '{raw_path}' from manifest '{manifest_path}'. "
        f"Tried: {candidates}"
    )


def parse_case_name(case_name: str) -> tuple[int, int, int | None] | None:
    m = CASE_RE.search(case_name)
    if not m:
        return None
    geo = int(m.group("geo"))
    run = int(m.group("run"))
    iter_raw = m.group("iter")
    iteration = int(iter_raw) if iter_raw is not None else None
    return geo, run, iteration


def find_z_cutoff(mask: np.ndarray, invalid_threshold: float = 0.95) -> int:
    z_layers = mask.shape[0]
    layer_size = mask.shape[1] * mask.shape[2]
    found_reservoir = False

    for z in range(z_layers):
        invalid_ratio = (layer_size - np.sum(mask[z])) / layer_size
        if invalid_ratio < invalid_threshold:
            found_reservoir = True
        if found_reservoir and invalid_ratio >= invalid_threshold:
            return z
    return z_layers


def build_log_perm_background(geology_h5: Path) -> np.ndarray:
    with h5py.File(geology_h5, "r") as h5:
        perm_x = h5["Input/PermX"][:].astype(np.float32)
        poro = h5["Input/Porosity"][:].astype(np.float32)

    valid = (poro > 1e-5) & (perm_x > -900.0)
    z_cutoff = find_z_cutoff(valid)
    if z_cutoff <= 0:
        z_cutoff = perm_x.shape[0]

    perm_crop = np.log10(np.maximum(perm_x[:z_cutoff], 1e-15))
    valid_crop = valid[:z_cutoff]

    z_slice = min(max(z_cutoff // 2, 0), z_cutoff - 1)
    bg = perm_crop[z_slice].T
    valid_bg = valid_crop[z_slice].T

    out = np.zeros_like(bg, dtype=np.float32)
    if np.any(valid_bg):
        vals = bg[valid_bg]
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax > vmin:
            out = (bg - vmin) / (vmax - vmin)
        out = np.clip(out, 0.0, 1.0)
    out[~valid_bg] = 0.0
    return out


def load_manifest_predictions(manifest_path: Path, repo_root: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    geology_info: dict[int, dict[str, Any]] = {}
    for g in manifest.get("geology_metadata", []):
        cfg = g.get("geology_config_id")
        if cfg is None:
            continue
        cfg_i = int(cfg)
        geology_info[cfg_i] = {
            "geology_name": g.get("geology_name", f"geo_{cfg_i}"),
            "geology_file": resolve_path(str(g.get("geology_file")), manifest_path, repo_root)
            if g.get("geology_file")
            else None,
        }

    rows: list[dict[str, Any]] = []
    run_to_wells: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for snap in manifest.get("snapshots", []):
        run_id = int(snap["run_id"])
        json_path = resolve_path(str(snap["json_path"]), manifest_path, repo_root)
        iteration = snap.get("iteration")

        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if iteration is None:
            iteration = payload.get("iteration")
        iter_val = int(iteration) if iteration is not None else 0
        run_to_wells[(run_id, iter_val)] = payload.get("wells", [])
        for pred in payload.get("predictions_by_geology", []):
            geo_id = int(pred["geology_config_id"])
            rows.append(
                {
                    "run_id": run_id,
                    "iteration": iter_val,
                    "geology_config_id": geo_id,
                    "pred_discounted_revenue": float(pred["discounted_total_revenue"]),
                    "pred_total_energy": float(pred["total_energy_production"]),
                    "snapshot_id": str(payload.get("snapshot_id", snap.get("snapshot_id", ""))),
                }
            )
            if geo_id not in geology_info:
                geology_info[geo_id] = {
                    "geology_name": str(pred.get("geology_name", f"geo_{geo_id}")),
                    "geology_file": resolve_path(str(pred.get("geology_file")), manifest_path, repo_root)
                    if pred.get("geology_file")
                    else None,
                }

    return rows, run_to_wells, geology_info




def load_intersect_actuals(h5_path: Path) -> dict[tuple[int, int, int | None], dict[str, float]]:
    out: dict[tuple[int, int, int | None], dict[str, float]] = {}

    with h5py.File(h5_path, "r") as h5:
        energy_price = float(h5.attrs.get("target_graph_discounted_net_revenue_energy_price_kwh", 0.0))
        discount_rate = resolve_real_discount_rate_from_attrs(h5.attrs)

        for case_name in h5.keys():
            parsed = parse_case_name(case_name)
            if parsed is None:
                continue
            geo, run, iteration = parsed
            group = h5[case_name]

            energy_total = float(group["field_energy_production_total"][-1])
            if "field_discounted_net_revenue" in group:
                discounted = float(group["field_discounted_net_revenue"][()])
            else:
                discounted = discounted_revenue_from_rates(
                    group["field_energy_production_rate"][:],
                    group["field_energy_injection_rate"][:],
                    energy_price,
                    discount_rate,
                )

            iter_val = iteration if iteration is not None else 0
            out[(geo, run, iter_val)] = {
                "actual_total_energy": energy_total,
                "actual_discounted_revenue": discounted,
                "case_name": case_name,
                "iteration": iter_val,
            }

    return out


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err)) if abs_err.size else float("nan")
    r2 = compute_r2(y_true, y_pred)

    # Percent errors are undefined when true value is ~0.
    eps = 1e-12
    pct_mask = np.abs(y_true) > eps
    if np.any(pct_mask):
        pct_err = 100.0 * (y_pred[pct_mask] - y_true[pct_mask]) / y_true[pct_mask]
        mape = float(np.mean(np.abs(pct_err)))
        mpe = float(np.mean(pct_err))
        pct_min = float(np.min(pct_err))
        pct_max = float(np.max(pct_err))
        pct_q25 = float(np.percentile(pct_err, 25))
        pct_q50 = float(np.percentile(pct_err, 50))
        pct_q75 = float(np.percentile(pct_err, 75))

        abs_pct_err = np.abs(pct_err)
        abs_pct_q25 = float(np.percentile(abs_pct_err, 25))
        abs_pct_q50 = float(np.percentile(abs_pct_err, 50))
        abs_pct_q75 = float(np.percentile(abs_pct_err, 75))
    else:
        mape = float("nan")
        mpe = float("nan")
        pct_min = float("nan")
        pct_max = float("nan")
        pct_q25 = float("nan")
        pct_q50 = float("nan")
        pct_q75 = float("nan")
        abs_pct_q25 = float("nan")
        abs_pct_q50 = float("nan")
        abs_pct_q75 = float("nan")

    abs_q25 = float(np.percentile(abs_err, 25))
    abs_q50 = float(np.percentile(abs_err, 50))
    abs_q75 = float(np.percentile(abs_err, 75))

    return {
        "mae": mae,
        "mape": mape,
        "mpe": mpe,
        "pct_err_min": pct_min,
        "pct_err_max": pct_max,
        "pct_err_q25": pct_q25,
        "pct_err_q50": pct_q50,
        "pct_err_q75": pct_q75,
        "abs_pct_err_q25": abs_pct_q25,
        "abs_pct_err_q50": abs_pct_q50,
        "abs_pct_err_q75": abs_pct_q75,
        "abs_err_q25": abs_q25,
        "abs_err_q50": abs_q50,
        "abs_err_q75": abs_q75,
        "r2": r2,
        "n": int(y_true.size),
    }


def _error_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    abs_err = np.abs(y_pred - y_true)

    eps = 1e-12
    pct_mask = np.abs(y_true) > eps
    pct_err = np.array([], dtype=np.float64)
    if np.any(pct_mask):
        pct_err = 100.0 * (y_pred[pct_mask] - y_true[pct_mask]) / y_true[pct_mask]

    return abs_err, pct_err


def style_ax(ax):
    ax.set_facecolor(MANIM_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(MANIM_WHITE)
    ax.tick_params(colors=MANIM_WHITE)
    ax.xaxis.label.set_color(MANIM_WHITE)
    ax.yaxis.label.set_color(MANIM_WHITE)
    ax.title.set_color(MANIM_WHITE)


def save_scatter_plot(
    rows: list[dict[str, Any]], geology_info: dict[int, dict[str, Any]], out_path: Path
) -> dict[str, dict[str, float]]:
    plt.rcParams.update(
        {
            "figure.facecolor": MANIM_BG,
            "axes.facecolor": MANIM_BG,
            "axes.edgecolor": MANIM_WHITE,
            "axes.labelcolor": MANIM_WHITE,
            "xtick.color": MANIM_WHITE,
            "ytick.color": MANIM_WHITE,
            "text.color": MANIM_WHITE,
            "legend.facecolor": "#111111",
            "legend.edgecolor": MANIM_GREY,
        }
    )

    geos = sorted({int(r["geology_config_id"]) for r in rows})
    cmap = plt.colormaps.get_cmap("tab10")
    geo_color = {g: cmap(i) for i, g in enumerate(geos)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=MANIM_BG)

    metric_specs = [
        (
            "pred_total_energy",
            "actual_total_energy",
            "Total Energy",
            "total_energy",
            axes[0],
        ),
        (
            "pred_discounted_revenue",
            "actual_discounted_revenue",
            "Discounted Revenue",
            "discounted_revenue",
            axes[1],
        ),
    ]

    metric_summaries: dict[str, dict[str, float]] = {}

    for pred_key, actual_key, label, metric_slug, ax in metric_specs:
        style_ax(ax)

        pred_all = np.array([float(r[pred_key]) for r in rows], dtype=np.float64)
        act_all = np.array([float(r[actual_key]) for r in rows], dtype=np.float64)

        for g in geos:
            sub = [r for r in rows if int(r["geology_config_id"]) == g]
            pred = np.array([float(r[pred_key]) for r in sub], dtype=np.float64)
            act = np.array([float(r[actual_key]) for r in sub], dtype=np.float64)
            geo_name = geology_info.get(g, {}).get("geology_name", f"geo_{g}")
            ax.scatter(
                pred,
                act,
                s=26,
                alpha=0.85,
                color=geo_color[g],
                edgecolors=MANIM_WHITE,
                linewidths=0.3,
                label=f"{geo_name} (cfg {g})",
            )

        lo = float(min(np.min(pred_all), np.min(act_all)))
        hi = float(max(np.max(pred_all), np.max(act_all)))
        margin = 0.03 * (hi - lo + 1e-12)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color=MANIM_GREY, linewidth=1.2)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)

        summary = compute_error_summary(act_all, pred_all)
        metric_summaries[metric_slug] = summary

        ax.set_xlabel(f"Surrogate Predicted {label}")
        ax.set_ylabel(f"Intersect Actual {label}")
        ax.set_title(
            (
                f"{label}: Predicted vs Actual\n"
                f"MAE={summary['mae']:.3e}, MAPE={summary['mape']:.2f}%, "
                f"Min%={summary['pct_err_min']:.2f}, Max%={summary['pct_err_max']:.2f}, "
                f"R2={summary['r2']:.4f}"
            )
        )
        ax.grid(True, linestyle="--", alpha=0.2, color=MANIM_GREY)

    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig)
    return metric_summaries


def save_error_distribution_plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": MANIM_BG,
            "axes.facecolor": MANIM_BG,
            "axes.edgecolor": MANIM_WHITE,
            "axes.labelcolor": MANIM_WHITE,
            "xtick.color": MANIM_WHITE,
            "ytick.color": MANIM_WHITE,
            "text.color": MANIM_WHITE,
            "legend.facecolor": "#111111",
            "legend.edgecolor": MANIM_GREY,
        }
    )

    specs = [
        ("pred_total_energy", "actual_total_energy", "Total Energy"),
        ("pred_discounted_revenue", "actual_discounted_revenue", "Discounted Revenue"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=MANIM_BG)

    for ridx, (pred_key, act_key, label) in enumerate(specs):
        y_pred = np.array([float(r[pred_key]) for r in rows], dtype=np.float64)
        y_true = np.array([float(r[act_key]) for r in rows], dtype=np.float64)
        abs_err, pct_err = _error_arrays(y_true, y_pred)

        ax_pct = axes[ridx, 0]
        style_ax(ax_pct)
        if pct_err.size > 0:
            ax_pct.hist(pct_err, bins=36, color=MANIM_BLUE, alpha=0.8)
            ax_pct.axvline(0.0, linestyle="--", color=MANIM_GREY, linewidth=1.2)
            ax_pct.axvline(np.mean(pct_err), linestyle="-", color=MANIM_ORANGE, linewidth=1.4)
            q1, q2, q3 = np.percentile(pct_err, [25, 50, 75])
            ax_pct.set_title(
                (
                    f"{label}: Signed Percent Error Distribution\n"
                    f"MPE={np.mean(pct_err):.2f}% | Q1={q1:.2f}% | Med={q2:.2f}% | Q3={q3:.2f}%"
                )
            )
        else:
            ax_pct.set_title(f"{label}: Signed Percent Error Distribution (no valid points)")
        ax_pct.set_xlabel("Percent Error (%) = 100*(pred-actual)/actual")
        ax_pct.set_ylabel("Count")
        ax_pct.grid(True, linestyle="--", alpha=0.2, color=MANIM_GREY)

        ax_abs = axes[ridx, 1]
        style_ax(ax_abs)
        if abs_err.size > 0:
            ax_abs.hist(abs_err, bins=36, color=MANIM_ORANGE, alpha=0.8)
            q1a, q2a, q3a = np.percentile(abs_err, [25, 50, 75])
            ax_abs.set_title(
                (
                    f"{label}: Absolute Error Distribution\n"
                    f"Q1={q1a:.3e} | Med={q2a:.3e} | Q3={q3a:.3e}"
                )
            )
        else:
            ax_abs.set_title(f"{label}: Absolute Error Distribution (no valid points)")
        ax_abs.set_xlabel("Absolute Error = |pred-actual|")
        ax_abs.set_ylabel("Count")
        ax_abs.grid(True, linestyle="--", alpha=0.2, color=MANIM_GREY)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig)


def save_geology_bias_plot(
    rows: list[dict[str, Any]], geology_info: dict[int, dict[str, Any]], out_path: Path
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": MANIM_BG,
            "axes.facecolor": MANIM_BG,
            "axes.edgecolor": MANIM_WHITE,
            "axes.labelcolor": MANIM_WHITE,
            "xtick.color": MANIM_WHITE,
            "ytick.color": MANIM_WHITE,
            "text.color": MANIM_WHITE,
            "legend.facecolor": "#111111",
            "legend.edgecolor": MANIM_GREY,
        }
    )

    geos = sorted({int(r["geology_config_id"]) for r in rows})
    cmap = plt.colormaps.get_cmap("tab10")
    geo_color = {g: cmap(i) for i, g in enumerate(geos)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=MANIM_BG)

    specs = [
        ("pred_total_energy", "actual_total_energy", "Total Energy"),
        ("pred_discounted_revenue", "actual_discounted_revenue", "Discounted Revenue"),
    ]

    for ax, (pred_key, act_key, label) in zip(axes, specs):
        style_ax(ax)
        ax.axvline(0.0, linestyle="--", color=MANIM_GREY, linewidth=1.1)

        # Overlay geology-specific signed percent-error distributions
        for g in geos:
            sub = [r for r in rows if int(r["geology_config_id"]) == g]
            y_true = np.array([float(r[act_key]) for r in sub], dtype=np.float64)
            y_pred = np.array([float(r[pred_key]) for r in sub], dtype=np.float64)
            _, pct_err = _error_arrays(y_true, y_pred)
            if pct_err.size == 0:
                continue

            geo_name = geology_info.get(g, {}).get("geology_name", f"geo_{g}")
            ax.hist(
                pct_err,
                bins=24,
                alpha=0.35,
                color=geo_color[g],
                label=f"{geo_name} (cfg {g})",
                edgecolor="none",
            )
            ax.axvline(
                float(np.mean(pct_err)),
                linestyle="-",
                linewidth=1.3,
                color=geo_color[g],
                alpha=0.95,
            )

        ax.set_title(f"{label}: Bias Distribution by Geology")
        ax.set_xlabel("Signed Percent Error (%) = 100*(pred-actual)/actual")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.2, color=MANIM_GREY)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig)


def save_iteration_trend_plot(
    rows: list[dict[str, Any]], geology_info: dict[int, dict[str, Any]], out_path: Path
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": MANIM_BG,
            "axes.facecolor": MANIM_BG,
            "axes.edgecolor": MANIM_WHITE,
            "axes.labelcolor": MANIM_WHITE,
            "xtick.color": MANIM_WHITE,
            "ytick.color": MANIM_WHITE,
            "text.color": MANIM_WHITE,
            "legend.facecolor": "#111111",
            "legend.edgecolor": MANIM_GREY,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 11), facecolor=MANIM_BG, sharex=True)

    specs = [
        (
            "pred_total_energy",
            "actual_total_energy",
            "Total Energy",
            axes[0],
        ),
        (
            "pred_discounted_revenue",
            "actual_discounted_revenue",
            "Discounted Revenue",
            axes[1],
        ),
    ]

    for pred_key, act_key, label, ax in specs:
        style_ax(ax)

        rng = np.random.default_rng(0)

        all_iters = sorted({int(r["iteration"]) for r in rows})
        runs = sorted({int(r["run_id"]) for r in rows})

        per_iter_surrogate: list[list[float]] = []
        per_iter_actual: list[list[float]] = []
        surrogate_avg = []
        actual_avg = []

        for iter_val in all_iters:
            run_means_s = []
            run_means_a = []
            for run_id in runs:
                run_rows = [
                    r
                    for r in rows
                    if int(r["iteration"]) == iter_val and int(r["run_id"]) == run_id
                ]
                if not run_rows:
                    continue
                run_means_s.append(float(np.mean([float(r[pred_key]) for r in run_rows])))
                run_means_a.append(float(np.mean([float(r[act_key]) for r in run_rows])))
            per_iter_surrogate.append(run_means_s)
            per_iter_actual.append(run_means_a)
            if run_means_s:
                surrogate_avg.append(float(np.mean(run_means_s)))
            else:
                surrogate_avg.append(float("nan"))
            if run_means_a:
                actual_avg.append(float(np.mean(run_means_a)))
            else:
                actual_avg.append(float("nan"))

        x = np.arange(len(all_iters), dtype=np.float64)
        pos_s = x - 0.18
        pos_a = x + 0.18

        viol_s = ax.violinplot(
            per_iter_surrogate,
            positions=pos_s,
            widths=0.28,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        viol_a = ax.violinplot(
            per_iter_actual,
            positions=pos_a,
            widths=0.28,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        for body in viol_s["bodies"]:
            body.set_facecolor(MANIM_WHITE)
            body.set_edgecolor(MANIM_WHITE)
            body.set_alpha(0.45)
        for body in viol_a["bodies"]:
            body.set_facecolor(MANIM_ORANGE)
            body.set_edgecolor(MANIM_ORANGE)
            body.set_alpha(0.45)
        for part in ("cmedians",):
            viol_s[part].set_color(MANIM_WHITE)
            viol_s[part].set_linewidth(1.5)
            viol_a[part].set_color(MANIM_ORANGE)
            viol_a[part].set_linewidth(1.5)

        jitter = 0.08
        for idx, iter_val in enumerate(all_iters):
            s_vals = per_iter_surrogate[idx]
            a_vals = per_iter_actual[idx]
            if s_vals:
                x_s = pos_s[idx] + rng.uniform(-jitter, jitter, size=len(s_vals))
                ax.scatter(
                    x_s,
                    s_vals,
                    color=MANIM_WHITE,
                    alpha=0.5,
                    s=18,
                    edgecolors=MANIM_GREY,
                    linewidths=0.3,
                )
            if a_vals:
                x_a = pos_a[idx] + rng.uniform(-jitter, jitter, size=len(a_vals))
                ax.scatter(
                    x_a,
                    a_vals,
                    color=MANIM_ORANGE,
                    alpha=0.5,
                    s=18,
                    edgecolors=MANIM_GREY,
                    linewidths=0.3,
                )

        ax.plot(x, surrogate_avg, color=MANIM_WHITE, linewidth=2.2, linestyle="-", label="Mean surrogate")
        ax.plot(x, actual_avg, color=MANIM_ORANGE, linewidth=2.2, linestyle="--", label="Mean intersect")

        ax.set_ylabel(label)
        ax.set_title(f"{label}: Surrogate vs Intersect by Iteration")
        ax.grid(True, linestyle="--", alpha=0.2, color=MANIM_GREY)

    axes[1].set_xlabel("Iteration")
    axes[1].set_xticks(np.arange(len(all_iters), dtype=np.float64))
    axes[1].set_xticklabels([str(i) for i in all_iters])
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    axes[1].legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
    plt.close(fig)


def draw_well_overlay(ax, bg: np.ndarray | None, wells: list[dict[str, Any]]) -> None:
    style_ax(ax)

    if bg is not None:
        im = ax.imshow(
            bg,
            origin="lower",
            cmap="viridis",
            alpha=0.58,
            extent=[0, bg.shape[1], 0, bg.shape[0]],
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors=MANIM_WHITE)
        cbar.set_label("Normalized Log PermX", fontsize=9, color=MANIM_WHITE)
        cbar.outline.set_edgecolor(MANIM_WHITE)

    for w in wells:
        is_inj = str(w.get("type", "")).lower() == "injector"
        marker = "^" if is_inj else "v"
        color = MANIM_BLUE if is_inj else MANIM_ORANGE
        ax.scatter(
            float(w.get("x", np.nan)),
            float(w.get("y", np.nan)),
            marker=marker,
            color=color,
            s=120,
            edgecolors=MANIM_WHITE,
            linewidths=1.0,
            alpha=0.9,
        )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")


def save_best_worst_overlays(
    rows: list[dict[str, Any]],
    geology_info: dict[int, dict[str, Any]],
    run_to_wells: dict[tuple[int, int], list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_cache: dict[int, np.ndarray | None] = {}
    for geo_id, info in geology_info.items():
        gfile = info.get("geology_file")
        if gfile is None:
            bg_cache[geo_id] = None
            continue
        try:
            bg_cache[geo_id] = build_log_perm_background(Path(gfile))
        except Exception:
            bg_cache[geo_id] = None

    metric_specs = [
        (
            "pred_total_energy",
            "actual_total_energy",
            "total_energy",
            "Total Energy",
        ),
        (
            "pred_discounted_revenue",
            "actual_discounted_revenue",
            "discounted_revenue",
            "Discounted Revenue",
        ),
    ]

    geos = sorted({int(r["geology_config_id"]) for r in rows})

    for pred_key, actual_key, slug, title in metric_specs:
        fig, axes = plt.subplots(
            len(geos),
            2,
            figsize=(14, max(5, 4.5 * len(geos))),
            facecolor=MANIM_BG,
            squeeze=False,
        )

        for ridx, geo in enumerate(geos):
            geo_rows = [r for r in rows if int(r["geology_config_id"]) == geo]
            errs = np.array(
                [abs(float(r[pred_key]) - float(r[actual_key])) for r in geo_rows],
                dtype=np.float64,
            )
            best = geo_rows[int(np.argmin(errs))]
            worst = geo_rows[int(np.argmax(errs))]

            geo_name = geology_info.get(geo, {}).get("geology_name", f"geo_{geo}")
            bg = bg_cache.get(geo)

            for cidx, (label, row) in enumerate((("Best", best), ("Worst", worst))):
                ax = axes[ridx, cidx]
                wells = run_to_wells.get(
                    (int(row["run_id"]), int(row.get("iteration", 0))), []
                )
                draw_well_overlay(ax, bg, wells)

                pred_val = float(row[pred_key])
                act_val = float(row[actual_key])
                err = abs(pred_val - act_val)
                ax.set_title(
                    (
                        f"{geo_name} (cfg {geo}) | {label} {title} Fit\n"
                        f"run={int(row['run_id']):04d}, iter={int(row.get('iteration', 0)):04d}, "
                        f"pred={pred_val:.3e}, "
                        f"actual={act_val:.3e}, |err|={err:.3e}"
                    ),
                    fontsize=10,
                    color=MANIM_WHITE,
                )

        fig.tight_layout()
        out_path = out_dir / f"best_worst_overlay_{slug}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=MANIM_BG)
        plt.close(fig)


def save_matched_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "iteration",
        "geology_config_id",
        "geology_name",
        "snapshot_id",
        "case_name",
        "pred_total_energy",
        "actual_total_energy",
        "abs_err_total_energy",
        "pred_discounted_revenue",
        "actual_discounted_revenue",
        "abs_err_discounted_revenue",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "run_id": int(r["run_id"]),
                    "iteration": int(r.get("iteration", 0)),
                    "geology_config_id": int(r["geology_config_id"]),
                    "geology_name": r.get("geology_name", ""),
                    "snapshot_id": r.get("snapshot_id", ""),
                    "case_name": r.get("case_name", ""),
                    "pred_total_energy": float(r["pred_total_energy"]),
                    "actual_total_energy": float(r["actual_total_energy"]),
                    "abs_err_total_energy": abs(
                        float(r["pred_total_energy"]) - float(r["actual_total_energy"])
                    ),
                    "pred_discounted_revenue": float(r["pred_discounted_revenue"]),
                    "actual_discounted_revenue": float(r["actual_discounted_revenue"]),
                    "abs_err_discounted_revenue": abs(
                        float(r["pred_discounted_revenue"]) - float(r["actual_discounted_revenue"])
                    ),
                }
            )


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    intersect_h5 = args.intersect_h5
    if not intersect_h5.is_absolute():
        intersect_h5 = (repo_root / intersect_h5).resolve()

    if args.snapshot_manifest is None:
        manifest_path = find_latest_manifest(repo_root)
    else:
        manifest_path = args.snapshot_manifest
        if not manifest_path.is_absolute():
            manifest_path = (repo_root / manifest_path).resolve()

    if not intersect_h5.exists():
        raise FileNotFoundError(f"Intersect H5 not found: {intersect_h5}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")

    pred_rows, run_to_wells, geology_info = load_manifest_predictions(manifest_path, repo_root)
    actuals = load_intersect_actuals(intersect_h5)

    matched: list[dict[str, Any]] = []
    missing_actual = 0
    for p in pred_rows:
        key = (
            int(p["geology_config_id"]),
            int(p["run_id"]),
            int(p.get("iteration", 0)) if p.get("iteration") is not None else 0,
        )
        act = actuals.get(key)
        if act is None:
            missing_actual += 1
            continue

        geo_id = int(p["geology_config_id"])
        geo_name = geology_info.get(geo_id, {}).get("geology_name", f"geo_{geo_id}")

        row = {
            **p,
            **act,
            "geology_name": geo_name,
        }
        matched.append(row)

    if not matched:
        raise RuntimeError(
            "No matched rows between surrogate snapshots and intersect_preprocessed.h5"
        )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_summaries = save_scatter_plot(
        matched, geology_info, output_dir / "scatter_surrogate_vs_intersect.png"
    )
    save_error_distribution_plot(
        matched, output_dir / "error_distribution_abs_vs_percent.png"
    )
    save_geology_bias_plot(
        matched, geology_info, output_dir / "bias_by_geology_percent_error.png"
    )
    save_iteration_trend_plot(
        matched, geology_info, output_dir / "iteration_surrogate_vs_intersect_trends.png"
    )
    save_best_worst_overlays(matched, geology_info, run_to_wells, output_dir)
    save_matched_csv(matched, output_dir / "matched_surrogate_vs_intersect.csv")

    print(f"Manifest: {manifest_path}")
    print(f"Intersect H5: {intersect_h5}")
    print(f"Matched rows: {len(matched)}")
    print(f"Missing actual rows: {missing_actual}")
    print("Metrics:")
    for metric_name, label in [
        ("total_energy", "Total Energy"),
        ("discounted_revenue", "Discounted Revenue"),
    ]:
        s = metric_summaries[metric_name]
        print(
            f"  - {label}: "
            f"MAPE={s['mape']:.4f}%, "
            f"Signed Mean % Error (MPE)={s['mpe']:.4f}%, "
            f"Percent Error [min={s['pct_err_min']:.4f}%, max={s['pct_err_max']:.4f}%], "
            f"Percent Error Quartiles [Q1={s['pct_err_q25']:.4f}%, Med={s['pct_err_q50']:.4f}%, Q3={s['pct_err_q75']:.4f}%], "
            f"Abs Percent Error Quartiles [Q1={s['abs_pct_err_q25']:.4f}%, Med={s['abs_pct_err_q50']:.4f}%, Q3={s['abs_pct_err_q75']:.4f}%], "
            f"Abs Error Quartiles [Q1={s['abs_err_q25']:.6e}, Med={s['abs_err_q50']:.6e}, Q3={s['abs_err_q75']:.6e}], "
            f"R2={s['r2']:.6f}, "
            f"MAE={s['mae']:.6e}, n={s['n']}"
        )
    print(f"Output dir: {output_dir}")
    print("Generated:")
    print(f"  - {output_dir / 'scatter_surrogate_vs_intersect.png'}")
    print(f"  - {output_dir / 'error_distribution_abs_vs_percent.png'}")
    print(f"  - {output_dir / 'bias_by_geology_percent_error.png'}")
    print(f"  - {output_dir / 'iteration_surrogate_vs_intersect_trends.png'}")
    print(f"  - {output_dir / 'best_worst_overlay_total_energy.png'}")
    print(f"  - {output_dir / 'best_worst_overlay_discounted_revenue.png'}")
    print(f"  - {output_dir / 'matched_surrogate_vs_intersect.csv'}")


if __name__ == "__main__":
    main()
