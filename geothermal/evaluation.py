"""Evaluation, metrics, and plotting utilities for the geothermal GNN."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from geothermal.model import EDGE_TYPES, HeteroGNNRegressor, TP_PROFILE_STATS
from geothermal.data import HeteroGraphScaler


# --------------- batch inference ---------------


def evaluate_split(
    model: HeteroGNNRegressor,
    graphs: list[HeteroData],
    scaler: HeteroGraphScaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    pred_scaled_parts: list[np.ndarray] = []
    true_scaled_parts: list[np.ndarray] = []
    case_ids: list[str] = []
    is_node_level = model.prediction_level == "node"
    filter_ext = getattr(graphs[0], "filter_extractors", True)
    should_filter = is_node_level and filter_ext

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            if should_filter:
                is_inj = batch["well"].is_injector.detach().cpu().numpy()
                ext_mask = is_inj < 0.5
                pred_scaled_parts.append(pred.detach().cpu().numpy()[ext_mask])
                true_scaled_parts.append(batch.y.detach().cpu().numpy()[ext_mask])
                for i in range(batch.num_graphs):
                    start = batch["well"].ptr[i].item()
                    end = batch["well"].ptr[i + 1].item()
                    n_ext = np.sum(ext_mask[start:end])
                    case_ids.extend([batch.case_id[i]] * int(n_ext))
            elif is_node_level:
                pred_scaled_parts.append(pred.detach().cpu().numpy())
                true_scaled_parts.append(batch.y.detach().cpu().numpy())
                for i in range(batch.num_graphs):
                    start = batch["well"].ptr[i].item()
                    end = batch["well"].ptr[i + 1].item()
                    n_nodes = end - start
                    case_ids.extend([batch.case_id[i]] * n_nodes)
            else:
                pred_scaled_parts.append(pred.detach().cpu().numpy())
                true_scaled_parts.append(batch.y.detach().cpu().numpy())
                case_ids.extend(list(batch.case_id))

    y_pred_scaled = np.concatenate(pred_scaled_parts)
    y_true_scaled = np.concatenate(true_scaled_parts)

    y_pred = scaler.inverse_targets(y_pred_scaled)
    y_true = scaler.inverse_targets(y_true_scaled)

    return y_true, y_pred, case_ids


# --------------- metrics ---------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt_flat = y_true.flatten()
    yp_flat = y_pred.flatten()

    mae = float(mean_absolute_error(yt_flat, yp_flat))
    rmse = float(np.sqrt(mean_squared_error(yt_flat, yp_flat)))
    medae = float(np.median(np.abs(yt_flat - yp_flat)))

    epsilon = np.finfo(np.float32).eps
    sig_mask = yt_flat > (np.mean(yt_flat) * 0.01)
    if np.any(sig_mask):
        mape_raw = (
            np.abs(
                (yt_flat[sig_mask] - yp_flat[sig_mask])
                / np.maximum(yt_flat[sig_mask], epsilon)
            )
            * 100.0
        )
        mape_clamped = np.clip(mape_raw, a_min=None, a_max=np.percentile(mape_raw, 99))
        mape = float(np.mean(mape_clamped))
    else:
        mape = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "mape": mape,
        "r2": float(r2_score(yt_flat, yp_flat)),
    }


# --------------- plotting helpers ---------------


def save_error_scatter_plots(
    output_dir: Path,
    split_data: dict[str, tuple[np.ndarray, np.ndarray]],
    target_label: str = "Target",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    alphas = {"train": 0.1, "val": 0.3, "test": 0.5}

    for split_name, (y_true, y_pred) in split_data.items():
        yt_flat = y_true.flatten()
        yp_flat = y_pred.flatten()
        residual = yp_flat - yt_flat

        p02_t, p98_t = np.percentile(yt_flat, [2, 98])
        p02_p, p98_p = np.percentile(yp_flat, [2, 98])
        c_min = min(p02_t, p02_p)
        c_max = max(p98_t, p98_p)

        c = colors.get(split_name, "tab:gray")
        a = alphas.get(split_name, 0.3)

        axes[0].scatter(yt_flat, yp_flat, s=2, alpha=a, color=c, label=split_name)

        pct_err = (residual / np.maximum(yt_flat, 1e-6)) * 100.0
        pct_err_clamped = np.clip(pct_err, -500, 500)
        axes[1].scatter(
            yt_flat, pct_err_clamped, s=2, alpha=a, color=c, label=split_name
        )

        if len(residual) > 1 and np.std(residual) > 1e-8:
            kde = stats.gaussian_kde(residual)
            eval_points = np.linspace(
                np.median(residual) - 3 * np.std(residual),
                np.median(residual) + 3 * np.std(residual),
                100,
            )
            axes[2].plot(
                eval_points, kde(eval_points), color=c, label=split_name, linewidth=2
            )
            axes[2].fill_between(eval_points, 0, kde(eval_points), color=c, alpha=0.1)

    axes[0].plot([c_min, c_max], [c_min, c_max], linestyle="--", color="red")
    axes[0].set_xlim(c_min, c_max)
    axes[0].set_ylim(c_min, c_max)
    axes[0].set_xlabel(f"Actual {target_label} (Core 96%)")
    axes[0].set_ylabel(f"Predicted {target_label}")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    axes[1].axhline(0.0, linestyle="--", color="red")
    axes[1].set_xlabel(f"Actual {target_label}")
    axes[1].set_ylabel("Percentage Error % (Clamped)")
    axes[1].set_title("Relative Error vs Actual")
    axes[1].legend()

    axes[2].axvline(0.0, linestyle="--", color="red")
    axes[2].set_xlabel("Absolute Error (Pred - Actual)")
    axes[2].set_ylabel("Probability Density")
    axes[2].set_title("KDE Error Distribution")
    axes[2].legend()

    fig.tight_layout()
    out_path = output_dir / "combined_prediction_error_dists.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def save_loss_curve_plot(metrics_csv_path: Path, output_path: Path) -> None:
    if not metrics_csv_path.exists():
        return

    hist: dict[str, dict[str, list[float]]] = {
        "loss": {"epochs": [], "train": [], "val": []},
        "mae_scaled": {"epochs": [], "train": [], "val": []},
    }

    with metrics_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_raw = row.get("epoch", "")
            if not epoch_raw:
                continue
            try:
                epoch = float(epoch_raw)
            except ValueError:
                continue

            for metric in ["loss", "mae_scaled"]:
                train_key = (
                    f"train_{metric}_epoch" if metric == "loss" else f"train_{metric}"
                )
                val_key = f"val_{metric}"

                t_val, v_val = row.get(train_key, ""), row.get(val_key, "")
                if t_val or v_val:
                    if epoch not in hist[metric]["epochs"]:
                        hist[metric]["epochs"].append(epoch)
                        hist[metric]["train"].append(float(t_val) if t_val else np.nan)
                        hist[metric]["val"].append(float(v_val) if v_val else np.nan)

    if not hist["loss"]["epochs"]:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["loss", "mae_scaled"]):
        eps = hist[metric]["epochs"]
        t_arr = np.array(hist[metric]["train"])
        v_arr = np.array(hist[metric]["val"])

        t_mask = ~np.isnan(t_arr)
        v_mask = ~np.isnan(v_arr)

        if np.any(t_mask):
            t_plot = t_arr[t_mask]
            if np.max(t_plot) > 10 * np.min(t_plot) and np.min(t_plot) > 0:
                ax.semilogy(np.array(eps)[t_mask], t_plot, label=f"Train {metric}")
            else:
                ax.plot(np.array(eps)[t_mask], t_plot, label=f"Train {metric}")

        if np.any(v_mask):
            v_plot = v_arr[v_mask]
            if np.max(v_plot) > 10 * np.min(v_plot) and np.min(v_plot) > 0:
                ax.semilogy(np.array(eps)[v_mask], v_plot, label=f"Val {metric}")
            else:
                ax.plot(np.array(eps)[v_mask], v_plot, label=f"Val {metric}")

        ax.set_xlabel("Epoch")
        ax.set_title(f"Curve: {metric.upper()}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved metric tracking curves: {output_path}")


def save_extreme_error_plots(
    output_dir: Path,
    split_name: str,
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(y_pred - y_true)

    mean_node_err = np.mean(abs_err, axis=1)

    k = min(top_k, len(mean_node_err))
    if k == 0:
        return

    high_idx = np.argsort(mean_node_err)[-k:][::-1]
    low_idx = np.argsort(mean_node_err)[:k]

    def _plot(indices: np.ndarray, title_suffix: str, filename: str) -> None:
        seen_labels = []
        clean_values = []
        for i in indices:
            seen_labels.append(f"{case_ids[i]}_node{i}")
            clean_values.append(mean_node_err[i])

        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(indices))))
        ypos = np.arange(len(indices))
        ax.barh(ypos, clean_values)
        ax.set_yticks(ypos)
        ax.set_yticklabels(seen_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Absolute error")
        ax.set_title(f"{split_name}: {title_suffix}")
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    _plot(
        high_idx,
        f"Top {k} highest-error cases",
        f"{split_name}_highest_error_cases.png",
    )
    _plot(
        low_idx, f"Top {k} lowest-error cases", f"{split_name}_lowest_error_cases.png"
    )


def save_extreme_error_graph_plots(
    output_dir: Path,
    split_name: str,
    graphs: list[HeteroData],
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_err = np.abs(y_pred - y_true)

    mean_node_err = np.mean(abs_err, axis=1)
    if len(mean_node_err) == 0:
        return

    highest_node_idx = int(np.argmax(mean_node_err))
    lowest_node_idx = int(np.argmin(mean_node_err))

    highest_case = case_ids[highest_node_idx]
    lowest_case = case_ids[lowest_node_idx]

    highest_idx = 0
    lowest_idx = 0
    for i, g in enumerate(graphs):
        if g.case_id == highest_case:
            highest_idx = i
        if g.case_id == lowest_case:
            lowest_idx = i

    relation_colors = {
        ("well", "inj_to_ext", "well"): "tab:purple",
        ("well", "inj_to_inj", "well"): "tab:orange",
        ("well", "ext_to_ext", "well"): "tab:green",
        ("well", "ext_to_inj", "well"): "tab:brown",
    }

    def _plot_case(idx: int, label: str, filename: str) -> None:
        graph = graphs[idx]
        pos_xy = graph["well"].pos_xy.detach().cpu().numpy()
        is_injector = graph["well"].is_injector.detach().cpu().numpy() > 0.5

        fig, ax = plt.subplots(figsize=(7, 7))

        for edge_type in EDGE_TYPES:
            edge_index = graph[edge_type].edge_index.detach().cpu().numpy()
            color = relation_colors[edge_type]
            if edge_index.shape[1] == 0:
                continue

            drawn = set()
            for src, dst in edge_index.T:
                key = tuple(sorted((int(src), int(dst))))
                if key in drawn:
                    continue
                drawn.add(key)
                x1, y1 = pos_xy[int(src)]
                x2, y2 = pos_xy[int(dst)]
                ax.plot([x1, x2], [y1, y2], color=color, alpha=0.35, linewidth=1.0)

        ax.scatter(
            pos_xy[is_injector, 0],
            pos_xy[is_injector, 1],
            c="tab:blue",
            s=40,
            label="Injector",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        ax.scatter(
            pos_xy[~is_injector, 0],
            pos_xy[~is_injector, 1],
            c="tab:red",
            s=40,
            label="Extractor",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

        rel_handles = [
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "inj_to_ext", "well")],
                lw=2,
                label="inj_to_ext",
            ),
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "inj_to_inj", "well")],
                lw=2,
                label="inj_to_inj",
            ),
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "ext_to_ext", "well")],
                lw=2,
                label="ext_to_ext",
            ),
            Line2D(
                [0],
                [0],
                color=relation_colors[("well", "ext_to_inj", "well")],
                lw=2,
                label="ext_to_inj",
            ),
        ]

        node_handles, node_labels = ax.get_legend_handles_labels()
        ax.legend(handles=node_handles + rel_handles, loc="best", fontsize=8)

        case_id = case_ids[idx]
        ax.set_title(f"{split_name} {label} error graph\n" f"case={case_id}")
        ax.set_xlabel("well x")
        ax.set_ylabel("well y")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        out_path = output_dir / filename
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved plot: {out_path}")

    _plot_case(highest_idx, "highest", f"{split_name}_highest_error_graph_xy.png")
    _plot_case(lowest_idx, "lowest", f"{split_name}_lowest_error_graph_xy.png")


def save_predictions_csv(
    output_path: Path,
    split_name: str,
    case_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_node_err = np.mean(np.abs(y_pred - y_true), axis=1)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "case_id", "sample_index", "mean_abs_error"])
        for i, (cid, ae) in enumerate(zip(case_ids, mean_node_err)):
            writer.writerow([split_name, cid, i, float(ae)])
    print(f"Saved predictions: {output_path}")
