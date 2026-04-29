#!/usr/bin/env python3
"""Ensemble inference with Adam optimization on a single geology."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import pickle
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Batch
from tqdm import tqdm

# Ensure repo-root modules are importable when running as `python inference/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compile_minimal_geothermal_h5 import (
    extract_well_data,
    build_wells_table,
    extract_vertical_profiles,
)
from preprocess_h5 import get_valid_mask, find_z_cutoff, PROPERTIES, PERM_PROPS
from geothermal.data import HeteroGraphScaler, build_single_hetero_data
from geothermal.model import HeteroGNNRegressor

# -----------------------------------------------------------------------------
# Plot styling (match run_differentiable_inference.py)
# -----------------------------------------------------------------------------
MANIM_BG = "#000000"
MANIM_BLUE = "#58C4DD"
MANIM_ORANGE = "#FF9000"
MANIM_WHITE = "#FFFFFF"
MANIM_GREY = "#888888"

FONT_SIZE = 18
TITLE_SIZE = 17
TICK_SIZE = 16
LEGEND_SIZE = 16
CBAR_LABEL_SIZE = 16


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
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


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(MANIM_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(MANIM_WHITE)
    ax.tick_params(colors=MANIM_WHITE, labelsize=TICK_SIZE)
    ax.xaxis.label.set_color(MANIM_WHITE)
    ax.yaxis.label.set_color(MANIM_WHITE)
    ax.title.set_color(MANIM_WHITE)


def _scaler_to_torch(
    scaler: HeteroGraphScaler, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(
        scaler.target_scaler.mean_, dtype=torch.float32, device=device
    )
    scale = torch.tensor(
        scaler.target_scaler.scale_, dtype=torch.float32, device=device
    )
    return mean, scale


def _inverse_target_torch(
    y_scaled: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return y_scaled * scale + mean


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _resolve_device(gpu_index: int) -> torch.device:
    if gpu_index >= 0 and torch.cuda.is_available():
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu {gpu_index} requested but only {torch.cuda.device_count()} "
                "GPU(s) are available."
            )
        return torch.device(f"cuda:{gpu_index}")
    return torch.device("cpu")


def _load_ensemble_members(
    ensemble_dir: Path,
    checkpoint_glob: str,
    scaler_filename: str,
) -> list[dict[str, Path]]:
    ckpt_paths = sorted(ensemble_dir.glob(checkpoint_glob))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found under {ensemble_dir} with '{checkpoint_glob}'."
        )
    members: list[dict[str, Path]] = []
    for ckpt_path in ckpt_paths:
        scaler_path = ckpt_path.parent / scaler_filename
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler: {scaler_path}")
        members.append({"checkpoint": ckpt_path, "scaler": scaler_path})
    return members


def _build_raw_graph(
    geology_path: Path,
    norm_config: dict,
    wells_cfg: list[dict],
) -> tuple[object, np.ndarray, tuple[int, int, int]]:
    with h5py.File(geology_path, "r") as src:
        valid_mask = get_valid_mask(src)
        z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
        valid_mask_cropped = valid_mask[:z_cutoff]

        physics_dict = {}
        for prop in PROPERTIES:
            data = src[f"Input/{prop}"][:z_cutoff].astype(np.float32)
            if prop in PERM_PROPS:
                data = np.log10(np.maximum(data, 1e-15))

            p_min = norm_config[prop]["min"]
            p_max = norm_config[prop]["max"]
            if p_max > p_min:
                normalized = (data - p_min) / (p_max - p_min)
            else:
                normalized = np.zeros_like(data)
            normalized = np.clip(normalized, 0.0, 1.0)
            normalized[~valid_mask_cropped] = 0.0
            physics_dict[prop] = torch.tensor(normalized, dtype=torch.float32)

        physics_dict["valid_mask"] = torch.tensor(
            valid_mask_cropped, dtype=torch.float32
        )
        full_shape = (z_cutoff, valid_mask.shape[1], valid_mask.shape[2])

        temp0_full = src["Input/Temperature0"][:]
        nx = full_shape[1]
        ny = full_shape[2]

        is_well = np.zeros((z_cutoff, nx, ny), dtype=np.int32)
        inj_rate = np.zeros((z_cutoff, nx, ny), dtype=np.float32)

        for w in wells_cfg:
            x_raw, y_raw = w["x"], w["y"]
            x = int(round(float(x_raw)))
            y = int(round(float(y_raw)))
            w_type = w.get("type", "injector").lower()
            if x < 0 or x >= nx or y < 0 or y >= ny:
                print(
                    f"Warning: Well at ({x_raw}, {y_raw}) is out of bounds "
                    f"for {geology_path.name} (Grid is {nx}x{ny})"
                )
                continue

            well_depth = min(int(w.get("depth", z_cutoff)), z_cutoff)
            for z in range(well_depth):
                if temp0_full[z, x, y] <= -900:
                    is_well[z, x, y] = -999
                    inj_rate[z, x, y] = -999
                else:
                    is_well[z, x, y] = 1
                    inj_rate[z, x, y] = 8000.0 if w_type == "injector" else -8000.0

        (
            x_idx,
            y_idx,
            depth,
            inj,
            perm_x,
            perm_y,
            perm_z,
            porosity,
            temp0,
            press0,
            depth_centroid,
        ) = extract_well_data(is_well, inj_rate, src)

        wells = build_wells_table(
            x_idx,
            y_idx,
            depth,
            inj,
            perm_x,
            perm_y,
            perm_z,
            porosity,
            temp0,
            press0,
        )
        vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)

    raw_graph = build_single_hetero_data(
        wells=wells,
        physics_dict=physics_dict,
        full_shape=full_shape,
        target="graph_discounted_net_revenue",
        target_val=0.0,
        vertical_profile=vertical_profiles,
        case_id="ensemble_inference",
    )

    perm_x_grid = physics_dict["PermX"].cpu().numpy()
    z_slice = perm_x_grid.shape[0] // 2
    background = perm_x_grid[z_slice, :, :].T

    return raw_graph, background, full_shape


def _save_coords_csv(
    output_dir: Path, coords_history: np.ndarray, well_count: int
) -> None:
    csv_path = output_dir / "coords_history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "well", "x", "y", "z"])
        for step in range(coords_history.shape[0]):
            for w in range(well_count):
                x, y, z = coords_history[step, w]
                writer.writerow([step, w, float(x), float(y), float(z)])
    print(f"Saved coords history: {csv_path}")


def _save_ensemble_csvs(
    output_dir: Path,
    predictions: np.ndarray,
    member_labels: list[str],
) -> None:
    pred_path = output_dir / "ensemble_predictions.csv"
    with pred_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + member_labels)
        for step in range(predictions.shape[0]):
            writer.writerow([step] + [float(x) for x in predictions[step]])
    print(f"Saved ensemble predictions: {pred_path}")

    mean_vals = predictions.mean(axis=1)
    std_vals = predictions.std(axis=1)
    summary_path = output_dir / "ensemble_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean", "std"])
        for step, (mean_v, std_v) in enumerate(zip(mean_vals, std_vals)):
            writer.writerow([step, float(mean_v), float(std_v)])
    print(f"Saved ensemble summary: {summary_path}")


def _plot_trajectory(
    output_dir: Path,
    coords_history: np.ndarray,
    background: np.ndarray,
    is_injector: np.ndarray,
    title: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=MANIM_BG)
    _style_ax(ax)
    ax.imshow(
        background,
        origin="lower",
        cmap="viridis",
        alpha=0.6,
        extent=[0, background.shape[1], 0, background.shape[0]],
    )

    for w in range(coords_history.shape[1]):
        is_inj = is_injector[w] > 0.5
        color = MANIM_BLUE if is_inj else MANIM_ORANGE
        ax.plot(
            coords_history[:, w, 0],
            coords_history[:, w, 1],
            color=color,
            linewidth=1.8,
            alpha=0.8,
        )
        ax.scatter(
            coords_history[0, w, 0],
            coords_history[0, w, 1],
            color=color,
            s=40,
            marker="o",
            edgecolors=MANIM_WHITE,
            linewidths=0.6,
            zorder=4,
        )
        ax.scatter(
            coords_history[-1, w, 0],
            coords_history[-1, w, 1],
            color=color,
            s=70,
            marker="^" if is_inj else "v",
            edgecolors=MANIM_WHITE,
            linewidths=0.6,
            zorder=5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(0, background.shape[0])
    fig.tight_layout()

    plot_path = output_dir / "trajectory.png"
    fig.savefig(plot_path, dpi=180, facecolor=MANIM_BG)
    plt.close(fig)
    print(f"Saved trajectory plot: {plot_path}")


def _plot_ensemble_traces(
    output_dir: Path,
    predictions: np.ndarray,
    member_labels: list[str],
) -> None:
    _set_plot_style()
    steps = np.arange(predictions.shape[0])
    mean_vals = predictions.mean(axis=1)
    std_vals = predictions.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=MANIM_BG)
    _style_ax(ax)
    for idx in range(predictions.shape[1]):
        ax.plot(
            steps,
            predictions[:, idx],
            color=MANIM_GREY,
            alpha=0.32,
            linewidth=1.0,
        )

    ax.plot(
        steps,
        mean_vals,
        color=MANIM_BLUE,
        linewidth=2.2,
        label="Ensemble mean",
    )
    ax.fill_between(
        steps,
        mean_vals - std_vals,
        mean_vals + std_vals,
        color=MANIM_BLUE,
        alpha=0.2,
        label="Mean ± std",
    )

    ax_std = ax.twinx()
    _style_ax(ax_std)
    ax_std.plot(
        steps,
        std_vals,
        color=MANIM_ORANGE,
        linewidth=1.8,
        label="Ensemble std",
    )
    ax_std.set_ylabel("Discounted net revenue std")

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Discounted net revenue")
    ax.set_title("Ensemble predictions along trajectory")
    ax.grid(True, linestyle="--", alpha=0.3, color=MANIM_GREY)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_std.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()

    plot_path = output_dir / "ensemble_traces.png"
    fig.savefig(plot_path, dpi=180, facecolor=MANIM_BG)
    plt.close(fig)
    print(f"Saved ensemble trace plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize well placement and evaluate an ensemble trajectory."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--optimization-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--optimization-objective",
        choices=["primary", "ensemble_mean", "ucb"],
        default=None,
        help="Objective for Adam optimization.",
    )
    parser.add_argument(
        "--ucb-beta",
        type=float,
        default=None,
        help="UCB beta (used when optimization-objective=ucb).",
    )
    args = parser.parse_args()

    config = _load_json(args.config)
    config_dir = args.config.resolve().parent

    geology_path = Path(config["geology_h5_file"]).expanduser()
    if not geology_path.is_absolute():
        geology_path = (config_dir / geology_path).resolve()

    norm_config_path = Path(config.get("norm_config", "norm_config.json"))
    if not norm_config_path.is_absolute():
        norm_config_path = (config_dir / norm_config_path).resolve()
    if not norm_config_path.exists():
        raise FileNotFoundError(f"Norm config not found: {norm_config_path}")
    norm_config = _load_json(norm_config_path)

    ensemble_dir = Path(config["ensemble_dir"]).expanduser()
    if not ensemble_dir.is_absolute():
        ensemble_dir = (config_dir / ensemble_dir).resolve()

    checkpoint_glob = config.get("checkpoint_glob", "run_*/checkpoints/best-*.ckpt")
    scaler_filename = config.get("scaler_filename", "scaler.pkl")
    primary_member = int(config.get("primary_member", 0))

    optimization_steps = (
        args.optimization_steps
        if args.optimization_steps is not None
        else int(config.get("optimization_steps", 10))
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(config.get("learning_rate", 0.5))
    )
    gpu_index = (
        args.gpu if args.gpu is not None else int(config.get("gpu", -1))
    )
    optimization_objective = (
        args.optimization_objective
        if args.optimization_objective is not None
        else config.get("optimization_objective", "primary")
    )
    ucb_beta = (
        args.ucb_beta
        if args.ucb_beta is not None
        else float(config.get("ucb_beta", 1.0))
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(config.get("output_dir", "inference/ensemble_outputs"))
    )
    if not output_dir.is_absolute():
        output_dir = (config_dir / output_dir).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = optimization_objective
    if optimization_objective == "ucb":
        tag = f"ucb_beta{ucb_beta:g}"
    output_dir = output_dir / f"run_{timestamp}_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = output_dir / "config.json"
    with config_copy_path.open("w") as f:
        json.dump(config, f, indent=2)

    members = _load_ensemble_members(ensemble_dir, checkpoint_glob, scaler_filename)
    if primary_member < 0 or primary_member >= len(members):
        raise ValueError(
            f"primary_member {primary_member} out of range for {len(members)} members."
        )

    wells_cfg = config["wells"]
    raw_graph, background, full_shape = _build_raw_graph(
        geology_path=geology_path,
        norm_config=norm_config,
        wells_cfg=wells_cfg,
    )

    device = _resolve_device(gpu_index)
    print(f"Using device: {device}")

    member_labels: list[str] = []
    member_payloads: list[dict[str, object]] = []

    # Preload all models and batches onto device to keep them in VRAM.
    for idx, member in enumerate(members):
        member_labels.append(f"member_{idx:02d}")
        with member["scaler"].open("rb") as f:
            member_scaler: HeteroGraphScaler = pickle.load(f)

        member_graph = member_scaler.transform_graph(raw_graph)
        member_batch = Batch.from_data_list([member_graph]).to(device)
        member_model = HeteroGNNRegressor.load_from_checkpoint(
            str(member["checkpoint"]), map_location=device
        ).to(device)
        member_model.eval()
        target_mean, target_scale = _scaler_to_torch(member_scaler, device)

        member_payloads.append(
            {
                "scaler": member_scaler,
                "batch": member_batch,
                "model": member_model,
                "target_mean": target_mean,
                "target_scale": target_scale,
            }
        )

    primary_payload = member_payloads[primary_member]
    primary_batch = primary_payload["batch"]
    coords = primary_batch["well"].pos_xyz.clone().detach().to(device)
    coords.requires_grad = True
    optimizer = optim.Adam([coords], lr=learning_rate)

    z_max, nx, ny = full_shape

    coords_history: list[np.ndarray] = []
    cached_predictions: list[np.ndarray] | None = None
    if optimization_objective in ("ensemble_mean", "ucb"):
        cached_predictions = []

    for step in tqdm(range(optimization_steps), desc="Optimize", unit="step"):
        coords_history.append(coords.detach().cpu().numpy())
        optimizer.zero_grad()

        if optimization_objective == "primary":
            primary_batch["well"].pos_xyz = coords
            predicted_scaled = primary_payload["model"](primary_batch).squeeze()
            loss = -predicted_scaled.mean()
        else:
            member_preds = []
            for payload in member_payloads:
                payload["batch"]["well"].pos_xyz = coords
                pred_scaled = payload["model"](payload["batch"]).view(-1)
                pred_unscaled = _inverse_target_torch(
                    pred_scaled,
                    payload["target_mean"],
                    payload["target_scale"],
                )
                member_preds.append(pred_unscaled.squeeze())
            preds_tensor = torch.stack(member_preds)
            mean_pred = preds_tensor.mean()
            std_pred = preds_tensor.std(unbiased=False)
            if optimization_objective == "ensemble_mean":
                objective = mean_pred
            else:
                objective = mean_pred + ucb_beta * std_pred
            loss = -objective
            if cached_predictions is not None:
                cached_predictions.append(
                    preds_tensor.detach().cpu().numpy().reshape(-1)
                )
        loss.backward()

        gradients = coords.grad
        with torch.no_grad():
            for d, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                mask_lower = (coords[:, d] <= 1e-4) & (gradients[:, d] > 0)
                mask_upper = (coords[:, d] >= max_val - 1e-4) & (gradients[:, d] < 0)
                mask_out = mask_lower | mask_upper
                if mask_out.any():
                    gradients[:, d][mask_out] = 0.0
                    if coords in optimizer.state and "exp_avg" in optimizer.state[coords]:
                        optimizer.state[coords]["exp_avg"][:, d][mask_out] = 0.0

        optimizer.step()
        with torch.no_grad():
            coords[:, 0].clamp_(0, nx - 1)
            coords[:, 1].clamp_(0, ny - 1)
            coords[:, 2].clamp_(0, z_max - 1)

    coords_history.append(coords.detach().cpu().numpy())
    if cached_predictions is not None:
        with torch.no_grad():
            member_preds = []
            for payload in member_payloads:
                payload["batch"]["well"].pos_xyz = coords
                pred_scaled = payload["model"](payload["batch"]).view(-1)
                pred_unscaled = _inverse_target_torch(
                    pred_scaled,
                    payload["target_mean"],
                    payload["target_scale"],
                )
                member_preds.append(pred_unscaled.squeeze())
            preds_tensor = torch.stack(member_preds)
            cached_predictions.append(
                preds_tensor.detach().cpu().numpy().reshape(-1)
            )
    coords_history_np = np.stack(coords_history, axis=0)

    if cached_predictions is not None:
        ensemble_predictions = np.stack(cached_predictions, axis=0)
    else:
        ensemble_predictions = np.zeros((coords_history_np.shape[0], len(members)))
        for step in tqdm(
            range(coords_history_np.shape[0]), desc="Ensemble eval", unit="step"
        ):
            coords_tensor = torch.tensor(
                coords_history_np[step],
                dtype=primary_batch["well"].pos_xyz.dtype,
                device=device,
            )
            for idx, payload in enumerate(member_payloads):
                member_batch = payload["batch"]
                member_model = payload["model"]
                with torch.no_grad():
                    member_batch["well"].pos_xyz = coords_tensor
                    pred_scaled = member_model(member_batch).view(-1)
                    pred_unscaled = _inverse_target_torch(
                        pred_scaled,
                        payload["target_mean"],
                        payload["target_scale"],
                    )
                ensemble_predictions[step, idx] = float(pred_unscaled.item())

    is_injector = primary_batch["well"].is_injector.cpu().numpy()

    _save_coords_csv(output_dir, coords_history_np, coords_history_np.shape[1])
    _save_ensemble_csvs(output_dir, ensemble_predictions, member_labels)

    _plot_trajectory(
        output_dir,
        coords_history_np,
        background,
        is_injector,
        title="Well trajectory on PermX background",
    )
    _plot_ensemble_traces(output_dir, ensemble_predictions, member_labels)


if __name__ == "__main__":
    main()
