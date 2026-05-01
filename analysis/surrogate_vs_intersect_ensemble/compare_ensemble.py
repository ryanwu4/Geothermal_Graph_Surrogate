#!/usr/bin/env python3
"""
Ensemble Surrogate vs INTERSECT Analysis

Generates plots comparing the performance of the ensemble surrogate against 
ground truth INTERSECT numerical simulation.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "black", 
    "figure.facecolor": "black", 
    "axes.grid": True, 
    "grid.color": "#333333", 
    "text.color": "white", 
    "axes.labelcolor": "white", 
    "xtick.color": "white", 
    "ytick.color": "white"
})

COLORS = {
    "primary": "#58C4DD",       # Manim Blue
    "ensemble_mean": "#FF9000", # Manim Orange
    "ucb": "#88C273",           # Soft Green
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intersect-h5",
        type=Path,
        required=True,
        help="Path to preprocessed Intersect output H5"
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        required=True,
        help="Directory containing the ensemble output folders with manifests"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/surrogate_vs_intersect_ensemble"),
        help="Directory for plots and matched CSV"
    )
    return parser.parse_args()


def parse_case_name(case_name: str):
    # e.g., v2.5_ensemble_primary_74_run0001_iter0050
    # or v2.5_ensemble_mean_...
    m = re.search(r"ensemble_(?P<obj>primary|mean|ucb|ensemble_mean)(?:_beta\d+)?_(?P<geo>\d+)_run(?P<run>\d+)_iter(?P<iter>\d+)", case_name)
    if not m:
        return None
    obj = m.group("obj")
    if obj == "mean":
        obj = "ensemble_mean"
    return obj, int(m.group("run")), int(m.group("iter"))

def load_intersect_actuals(h5_path: Path):
    out = {}
    with h5py.File(h5_path, "r") as h5:
        for case_name in h5.keys():
            parsed = parse_case_name(case_name)
            if not parsed:
                continue
            obj, run, iteration = parsed
            group = h5[case_name]
            
            if "field_discounted_net_revenue" in group:
                discounted = float(group["field_discounted_net_revenue"][()])
            else:
                logging.warning(f"Missing field_discounted_net_revenue in {case_name}")
                continue
                
            out[(obj, run, iteration)] = {
                "intersect_revenue": discounted,
                "case_name": case_name
            }
    return out

def determine_objective_from_manifest_path(manifest_path: Path):
    path_str = str(manifest_path)
    if "primary" in path_str:
        return "primary"
    elif "mean" in path_str:
        return "ensemble_mean"
    elif "ucb" in path_str:
        return "ucb"
    return "unknown"

def load_surrogate_predictions(manifests_dir: Path):
    manifests = list(manifests_dir.glob("**/snapshot_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifests found in {manifests_dir}")
        
    out = {}
    for mf in manifests:
        obj = determine_objective_from_manifest_path(mf)
        with mf.open("r") as f:
            data = json.load(f)
            
        for snap in data.get("snapshots", []):
            run = int(snap["run_id"])
            iteration = int(snap["iteration"])
            mean_rev = float(snap.get("mean_discounted_total_revenue", np.nan))
            std_rev = float(snap.get("std_discounted_total_revenue", 0.0))
            
            out[(obj, run, iteration)] = {
                "surrogate_mean": mean_rev,
                "surrogate_std": std_rev
            }
    return out

def build_merged_dataframe(intersect_data, surrogate_data):
    rows = []
    for key, idata in intersect_data.items():
        obj, run, iteration = key
        sdata = surrogate_data.get(key, {})
        if not sdata:
            logging.warning(f"Missing surrogate data for {key}")
            continue
            
        rows.append({
            "objective": obj,
            "run": run,
            "iteration": iteration,
            "intersect_revenue": idata["intersect_revenue"],
            "surrogate_mean": sdata["surrogate_mean"],
            "surrogate_std": sdata["surrogate_std"],
            "error": sdata["surrogate_mean"] - idata["intersect_revenue"],
            "percent_error": (sdata["surrogate_mean"] - idata["intersect_revenue"]) / idata["intersect_revenue"] * 100,
            "abs_error": abs(sdata["surrogate_mean"] - idata["intersect_revenue"])
        })
    return pd.DataFrame(rows)


def plot_violins_progression(df: pd.DataFrame, out_dir: Path, trendline=False):
    target_iters = [0, 50, 100]
    sub_df = df[df["iteration"].isin(target_iters)].copy()
    if sub_df.empty:
        logging.warning("No data found for iterations 0, 50, 100.")
        return
        
    plt.figure(figsize=(10, 6))
    
    ax = sns.violinplot(
        data=sub_df, 
        x="iteration", 
        y="intersect_revenue", 
        hue="objective",
        palette=COLORS,
        inner=None,
        alpha=0.6,
        linewidth=1
    )
    
    sns.stripplot(
        data=sub_df, 
        x="iteration", 
        y="intersect_revenue", 
        hue="objective",
        palette=COLORS,
        dodge=True,
        edgecolor="white",
        linewidth=1,
        alpha=0.8,
        ax=ax,
        legend=False
    )
    
    if trendline:
        means = sub_df.groupby(["iteration", "objective"])["intersect_revenue"].mean().reset_index()
        hue_order = ax.get_legend_handles_labels()[1]
        if not hue_order:
            hue_order = list(COLORS.keys())
            
        n_hues = len(hue_order)
        dodge_width = 0.8
        
        for hue_idx, obj in enumerate(hue_order):
            obj_means = means[means["objective"] == obj].sort_values("iteration")
            if obj_means.empty:
                continue
                
            x_coords = []
            y_coords = []
            for i, iter_val in enumerate(target_iters):
                row = obj_means[obj_means["iteration"] == iter_val]
                if not row.empty:
                    base_x = i
                    offset = (hue_idx - (n_hues - 1) / 2) * (dodge_width / n_hues)
                    x_coords.append(base_x + offset)
                    y_coords.append(row["intersect_revenue"].values[0])
                    
            plt.plot(x_coords, y_coords, marker='D', markersize=8, color=COLORS.get(obj, "white"), 
                     linestyle='-', linewidth=2, markeredgecolor='white', label=f"{obj} (Mean Intersect Revenue)")
                     
    plt.title("INTERSECT Discounted Revenue Progression", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Discounted Revenue ($)", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Objective", bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    suffix = "_with_trendline" if trendline else ""
    plt.savefig(out_dir / f"violin_progression{suffix}.png", dpi=200, bbox_inches="tight", facecolor='black')
    plt.close()

def plot_scatter_parity(df: pd.DataFrame, out_dir: Path, iteration=None):
    if iteration is not None:
        sub_df = df[df["iteration"] == iteration].copy()
        suffix = f"_iter{iteration}"
        title = f"Surrogate vs Intersect (Iteration {iteration})"
    else:
        sub_df = df.copy()
        suffix = "_all_iters"
        title = "Surrogate vs Intersect (All Iterations)"
        
    if sub_df.empty:
        return
        
    plt.figure(figsize=(8, 8))
    
    for obj, group in sub_df.groupby("objective"):
        plt.errorbar(
            group["surrogate_mean"], 
            group["intersect_revenue"],
            xerr=group["surrogate_std"],
            fmt='o',
            color=COLORS.get(obj, "k"),
            alpha=0.7,
            label=obj,
            markeredgecolor='white'
        )
        
    x_min = (sub_df["surrogate_mean"] - sub_df["surrogate_std"]).min()
    x_max = (sub_df["surrogate_mean"] + sub_df["surrogate_std"]).max()
    x_pad = (x_max - x_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    
    y_min = sub_df["intersect_revenue"].min()
    y_max = sub_df["intersect_revenue"].max()
    y_pad = (y_max - y_min) * 0.1
    y_min -= y_pad
    y_max += y_pad
    
    # Plot parity line spanning across the entire visible area
    global_min = min(x_min, y_min)
    global_max = max(x_max, y_max)
    plt.plot([global_min, global_max], [global_min, global_max], color='white', linestyle='--', alpha=0.5, label="1:1 Parity")
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title, fontsize=14)
    plt.xlabel("Surrogate Mean Predicted Revenue ($)", fontsize=12)
    plt.ylabel("INTERSECT Computed Revenue ($)", fontsize=12)
    plt.legend(facecolor='black', edgecolor='white')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_parity{suffix}.png", dpi=200, facecolor='black')
    plt.close()

def plot_error_histogram(df: pd.DataFrame, out_dir: Path, iteration=None):
    if iteration is not None:
        sub_df = df[df["iteration"] == iteration].copy()
        suffix = f"_iter{iteration}"
        title_base = f"Surrogate Error Distribution (Iter {iteration})"
    else:
        sub_df = df.copy()
        suffix = "_all_iters"
        title_base = "Surrogate Error Distribution (All Iters)"
        
    if sub_df.empty:
        return

    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        data=sub_df,
        x="percent_error",
        hue="objective",
        palette=COLORS,
        element="step",
        stat="count",
        common_norm=False,
        alpha=0.4,
        linewidth=2
    )
    
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    
    title_lines = [title_base]
    for obj in sorted(sub_df["objective"].unique()):
        sub = sub_df[sub_df["objective"] == obj]["percent_error"]
        if not sub.empty:
            q1, med, q3 = np.percentile(sub, [25, 50, 75])
            mape = np.mean(np.abs(sub))
            title_lines.append(f"{obj}: MAPE={mape:.2f}%, Q1={q1:.2f}%, Med={med:.2f}%, Q3={q3:.2f}%")
            
    plt.title("\n".join(title_lines), fontsize=12)
    plt.xlabel("Prediction Error (%)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"error_histogram{suffix}.png", dpi=200, facecolor='black')
    plt.close()

def plot_revenue_histogram(df: pd.DataFrame, out_dir: Path):
    sub_df = df[df["iteration"] == 100].copy()
    if sub_df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    sns.histplot(
        data=sub_df,
        x="intersect_revenue",
        hue="objective",
        palette=COLORS,
        element="step",
        stat="count",
        common_norm=False,
        alpha=0.4,
        linewidth=2
    )
    
    title_lines = ["INTERSECT Discounted Revenue Distribution (Iteration 100)"]
    for obj in sorted(sub_df["objective"].unique()):
        sub = sub_df[sub_df["objective"] == obj]["intersect_revenue"]
        if not sub.empty:
            mean_val = np.mean(sub)
            plt.axvline(mean_val, color=COLORS.get(obj, "white"), linestyle='--', alpha=0.8, linewidth=2)
            title_lines.append(f"{obj} Mean: ${mean_val:,.0f}")
            
    plt.title("\n".join(title_lines), fontsize=12)
    plt.xlabel("Discounted Revenue ($)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_dir / "revenue_histogram.png", dpi=200, facecolor='black')
    plt.close()

def plot_trendlines_only(df: pd.DataFrame, out_dir: Path):
    target_iters = [0, 50, 100]
    sub_df = df[df["iteration"].isin(target_iters)].copy()
    if sub_df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Plot individual run lines (INTERSECT revenue) faintly
    for obj, obj_group in sub_df.groupby("objective"):
        for run, run_group in obj_group.groupby("run"):
            run_group = run_group.sort_values("iteration")
            plt.plot(run_group["iteration"], run_group["intersect_revenue"], 
                     color=COLORS.get(obj, "white"), alpha=0.2, linewidth=1)
                     
    # Plot mean INTERSECT revenue lines thickly with error bars representing std across runs
    means = sub_df.groupby(["iteration", "objective"])["intersect_revenue"].agg(["mean", "std"]).reset_index()
    for obj, obj_means in means.groupby("objective"):
        obj_means = obj_means.sort_values("iteration")
        plt.errorbar(obj_means["iteration"], obj_means["mean"],
                     yerr=obj_means["std"],
                     color=COLORS.get(obj, "white"), marker='D', markersize=8,
                     linestyle='-', linewidth=3, markeredgecolor='white', capsize=5,
                     label=f"{obj} (Mean Intersect Revenue)")
                 
    y_min = (means["mean"] - means["std"]).min()
    y_max = (means["mean"] + means["std"]).max()
    y_pad = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_pad, y_max + y_pad)
                 
    plt.title("Revenue Progression (Individual INTERSECT Runs vs Mean INTERSECT)", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Discounted Revenue ($)", fontsize=12)
    plt.xticks(target_iters)
    
    plt.legend(facecolor='black', edgecolor='white')
    plt.tight_layout()
    plt.savefig(out_dir / "revenue_trendlines_only.png", dpi=200, facecolor='black')
    plt.close()

def plot_uncertainty_vs_error(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(
        data=df,
        x="surrogate_std",
        y="abs_error",
        hue="objective",
        palette=COLORS,
        alpha=0.8,
        edgecolor="white"
    )
    
    plt.title("Surrogate Uncertainty vs Absolute Error", fontsize=14)
    plt.xlabel("Surrogate Standard Deviation ($)", fontsize=12)
    plt.ylabel("Absolute Prediction Error ($)", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_dir / "uncertainty_vs_error.png", dpi=200, facecolor='black')
    plt.close()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading Intersect data from {args.intersect_h5}")
    intersect_data = load_intersect_actuals(args.intersect_h5)
    logging.info(f"Loaded {len(intersect_data)} intersect cases.")
    
    logging.info(f"Loading Surrogate data from {args.manifests_dir}")
    surrogate_data = load_surrogate_predictions(args.manifests_dir)
    logging.info(f"Loaded {len(surrogate_data)} surrogate predictions.")
    
    df = build_merged_dataframe(intersect_data, surrogate_data)
    logging.info(f"Merged Dataframe has {len(df)} rows.")
    
    if df.empty:
        logging.error("Merged dataframe is empty. Check case name matching logic.")
        return
        
    csv_path = args.output_dir / "matched_results.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved merged data to {csv_path}")
    
    logging.info("Generating Violin Progression Plot...")
    plot_violins_progression(df, args.output_dir, trendline=False)
    
    logging.info("Generating Violin Progression Plot with Trendline...")
    plot_violins_progression(df, args.output_dir, trendline=True)
    
    logging.info("Generating Scatter Parity Plots...")
    plot_scatter_parity(df, args.output_dir, iteration=None)
    plot_scatter_parity(df, args.output_dir, iteration=0)
    plot_scatter_parity(df, args.output_dir, iteration=100)
    
    logging.info("Generating Trendlines Plot...")
    plot_trendlines_only(df, args.output_dir)
    
    logging.info("Generating Error Histograms...")
    plot_error_histogram(df, args.output_dir, iteration=None)
    plot_error_histogram(df, args.output_dir, iteration=0)
    plot_error_histogram(df, args.output_dir, iteration=100)
    
    logging.info("Generating Revenue Histogram...")
    plot_revenue_histogram(df, args.output_dir)
    
    logging.info("Generating Uncertainty vs Error Plot...")
    plot_uncertainty_vs_error(df, args.output_dir)
    
    logging.info("All analysis complete!")

if __name__ == "__main__":
    main()
