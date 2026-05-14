#!/usr/bin/env python3
"""Train the heterogeneous GNN for geothermal energy prediction.

Usage examples:
    # Basic training
    python train.py --h5-path data_test/minimal_compiled_tp.h5

    # With top-10% withholding
    python train.py --h5-path data_test/minimal_compiled_tp.h5 --withhold-top-pct 10

    # With ablation
    python train.py --h5-path data/compiled.h5 --ablate vertical_profile edge_thermo
"""
from __future__ import annotations

import os
import argparse
import os
import json
import pickle
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from geothermal.data import (
    ABLATION_GROUPS,
    HeteroGraphScaler,
    apply_ablation,
    load_hetero_graphs,
    split_indices_stratified,
    withhold_top_pct,
)
from geothermal.evaluation import (
    compute_metrics,
    evaluate_split,
    save_error_scatter_plots,
    save_extreme_error_graph_plots,
    save_extreme_error_plots,
    save_loss_curve_plot,
    save_predictions_csv,
)
from geothermal.model import (
    HeteroGNNRegressor,
    TP_PROFILE_STATS,
    seed_all,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hetero GNN with typed edge relations and distance edge features."
    )
    parser.add_argument(
        "--h5-path", type=Path, default=Path("data_test/minimal_compiled_optimized.h5")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for data split and ensemble seeding.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Seed for train/val/test split. Defaults to --seed to keep the split fixed "
            "across ensemble members."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help=(
            "Ensemble member index. This offsets the base seed so each run has distinct "
            "initialization and shuffling."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("lightning_logs"),
        help="Root directory for per-run logs, checkpoints, and plots.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--early-stop-patience", type=int, default=60)

    default_workers = min(8, os.cpu_count() or 1) if hasattr(os, "cpu_count") else 4
    parser.add_argument("--num-workers", type=int, default=default_workers)

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--pooling",
        choices=["mean", "avg", "sum", "concat_sum_mean_max"],
        default="mean",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--no-whiten", action="store_true")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--stratified-split", action="store_true")
    parser.add_argument("--grad-clip-val", type=float, default=1.0)
    parser.add_argument("--extreme-k", type=int, default=20)
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument(
        "--ablate",
        nargs="+",
        default=[],
        metavar="GROUP",
        help=(
            "Feature groups to ablate (zero out). Available: "
            + ", ".join(sorted(ABLATION_GROUPS.keys()))
        ),
    )
    parser.add_argument(
        "--target",
        choices=[
            "node_wept",
            "node_tp_final",
            "graph_energy_total",
            "graph_energy_rate",
            "graph_discounted_net_revenue",
        ],
        default="graph_energy_total",
        help=(
            "Prediction target: node-level WEPT, node-level next-timestep T/P, "
            "graph-level energy, or graph-level discounted net revenue."
        ),
    )
    parser.add_argument(
        "--withhold-top-pct",
        type=float,
        default=0.0,
        help=(
            "Withhold the top N%% of datapoints (by the selected target) "
            "from training. Withheld run IDs are saved alongside the plots. "
            "Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--cache-to-gpu",
        action="store_true",
        help="If set, pushes the full dataset array into VRAM at startup. (Overrides num-workers to 0).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU indices to use (e.g., '0' or '0,1'). 'auto' means all available GPUs. Default: auto.",
    )
    parser.add_argument(
        "--edge-encoder",
        choices=["cnn", "svd"],
        default="cnn",
        help="Model to use for geometric edge encoding.",
    )
    parser.add_argument(
        "--svd-weights-path",
        type=Path,
        default=None,
        help="Path to precomputed SVD weights if edge-encoder is svd.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Path to a Lightning .ckpt to warm-start from. When set, the "
            "scaler from --scaler-path is reused and the model is initialized "
            "via load_from_checkpoint instead of fresh weights."
        ),
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=None,
        help=(
            "Pickle path of the HeteroGraphScaler to reuse with --checkpoint-path. "
            "Required when --checkpoint-path is set; ensures inference and training "
            "share the same normalization. Defaults to <ckpt-parent>/scaler.pkl."
        ),
    )
    parser.add_argument(
        "--max-epochs-finetune",
        type=int,
        default=None,
        help=(
            "If --checkpoint-path is set, override --max-epochs with this lower "
            "cap for finetuning. Ignored when training from scratch."
        ),
    )
    parser.add_argument(
        "--node-encoder",
        choices=["profile", "cnn", "hybrid"],
        default="cnn",
        help=(
            "How per-well node embeddings are produced. 'profile' uses the 25-d "
            "hand-coded well_vertical_profile (legacy, 33-d node features). 'cnn' "
            "replaces the profile with a learned 3D CNN over a physics slab around "
            "each well (9-d base scalars + latent_node_dim from the CNN). 'hybrid' "
            "keeps the 33-d profile AND concatenates the CNN embedding (33 + latent)."
        ),
    )
    parser.add_argument(
        "--edge-norm",
        choices=["batchnorm", "groupnorm"],
        default="groupnorm",
        help=(
            "Normalization layer inside the edge PhysicsSlabCNN. Default 'batchnorm' "
            "matches the existing behaviour. 'groupnorm' preserves per-instance "
            "absolute-scale signal (analogous to the v3 node-CNN fix)."
        ),
    )
    parser.add_argument(
        "--edge-raw-means",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set, the edge CNN MLP receives a 16-d projection of per-channel slab "
            "means computed pre-CNN (never normalized). Analogous to the node-CNN raw-means "
            "bypass; targets the same absolute-scale loss BatchNorm would otherwise cause. "
            "(use --no-edge-raw-means to disable)"
        ),
    )
    parser.add_argument(
        "--node-pad",
        type=int,
        default=3,
        help="XY pad (P) for per-node slab when --node-encoder cnn; slab XY extent is 2P+1.",
    )
    parser.add_argument(
        "--latent-node-dim",
        type=int,
        default=32,
        help="Dimension of the per-node CNN embedding when --node-encoder cnn.",
    )
    parser.add_argument(
        "--node-z-extent",
        choices=["full", "perforation"],
        default="full",
        help=(
            "Z range of the per-node slab. 'full' covers the active reservoir for every "
            "well (consistent physical Z resolution; perforation mask carries per-well info). "
            "'perforation' resamples only the well's perforation cells (variable physical "
            "resolution; legacy behaviour)."
        ),
    )
    parser.add_argument(
        "--enrich-global-attr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use an 8-d global_attr including reservoir-mean physics stats (PermX/Y/Z, "
            "Porosity, T0, P0 means + anisotropy ratio) instead of just n_wells. Gives "
            "the head a direct geology fingerprint without going through the GNN. "
            "(use --no-enrich-global-attr to disable)"
        ),
    )
    parser.add_argument(
        "--node-aggr",
        choices=["mean", "sum"],
        default="mean",
        help="NNConv aggregation operator. 'sum' preserves absolute message magnitudes.",
    )
    parser.add_argument(
        "--cnn-activation",
        choices=["relu", "gelu"],
        default="relu",
        help="Activation inside the slab CNNs (edge + node). Default ReLU matches existing.",
    )
    parser.add_argument(
        "--head-no-norm",
        action="store_true",
        help="Drop the LayerNorm layers inside the graph-prediction head.",
    )
    args = parser.parse_args()

    if args.checkpoint_path is not None and args.scaler_path is None:
        # Default to the scaler colocated with the checkpoint, matching how
        # train.py saves it (see save block below).
        args.scaler_path = args.checkpoint_path.parent / "scaler.pkl"

    prediction_level = (
        "node" if args.target in ("node_wept", "node_tp_final") else "graph"
    )
    output_dim = TP_PROFILE_STATS if args.target == "node_tp_final" else 1

    split_seed = args.seed if args.split_seed is None else args.split_seed
    run_seed = args.seed + args.run_id

    # Use run_seed for model initialization, data loader shuffling, and other RNG.
    L.seed_everything(run_seed, workers=True)
    seed_all(run_seed)
    # CRITICAL: MaxPool3d backward pass has NO deterministic CUDA implementation in PyTorch.
    # Lightning's seed_everything(workers=True) strictly forces determinism causing a crash.
    torch.use_deterministic_algorithms(False, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Enable TF32 for matmuls on Ampere+ GPUs (~2x throughput, no accuracy hit on
    # this surrogate). Lightning prints a warning when this is unset.
    torch.set_float32_matmul_precision("high")

    graphs_raw, targets = load_hetero_graphs(
        args.h5_path, target=args.target, node_encoder=args.node_encoder,
        enrich_global_attr=args.enrich_global_attr,
    )

    print(f"Prediction mode: {args.target} (level={prediction_level})")

    # --- Top-k% withholding ---
    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(
        save_dir=str(args.output_root),
        name="geothermal_hetero_gnn",
        version=f"run_{args.run_id:02d}",
    )
    if args.plots_dir is None:
        plots_dir = Path(logger.log_dir) / "plots"
    else:
        plots_subdir = args.plots_dir
        if plots_subdir.is_absolute():
            plots_subdir = Path(plots_subdir.name)
        plots_dir = Path(logger.log_dir) / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.withhold_top_pct > 0:
        graphs_raw, targets = withhold_top_pct(
            graphs_raw, targets, args.withhold_top_pct, plots_dir
        )

    if args.ablate:
        print(f"Ablation study: removing {args.ablate}")
        apply_ablation(graphs_raw, args.ablate)

    if args.stratified_split:
        # Try to load the case_id -> geology_index map adjacent to the H5 to
        # enable geology-aware stratification. Falls back to target-only.
        geology_indices = None
        geo_map_candidates = [
            args.h5_path.parent / "case_geology_map.json",
        ]
        for cand in geo_map_candidates:
            if cand.exists():
                try:
                    with open(cand) as f:
                        gmap = json.load(f)
                    case_ids = [g.case_id for g in graphs_raw]
                    if all(cid in gmap for cid in case_ids):
                        geology_indices = np.array(
                            [int(gmap[cid]["geology_index"]) for cid in case_ids],
                            dtype=np.int64,
                        )
                        print(f"Loaded geology map for stratification from {cand}")
                    else:
                        n_miss = sum(1 for cid in case_ids if cid not in gmap)
                        print(f"WARN: geology map at {cand} missing {n_miss}/{len(case_ids)} case_ids; falling back to target-only stratification")
                except Exception as e:
                    print(f"WARN: could not load {cand}: {e}; falling back to target-only stratification")
                break
        train_idx, val_idx, test_idx = split_indices_stratified(
            targets=targets,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=split_seed,
            geology_indices=geology_indices,
        )
    else:
        train_val_idx, test_idx = train_test_split(
            range(len(graphs_raw)),
            test_size=args.test_fraction,
            random_state=split_seed,
            shuffle=True,  # Keep shuffle=True for non-stratified
        )

        val_size_relative = args.val_fraction / (1.0 - args.test_fraction)

        # No stratification for train_val_strata in this 'else' block
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_relative,
            random_state=split_seed,
            shuffle=True,  # Keep shuffle=True for non-stratified
        )

    train_graphs_raw = [graphs_raw[i] for i in train_idx]
    val_graphs_raw = [graphs_raw[i] for i in val_idx]
    test_graphs_raw = [graphs_raw[i] for i in test_idx]

    if args.scaler_path is not None and args.checkpoint_path is not None:
        # Warm-start: reuse the prior iteration's scaler so the new model sees
        # inputs in the same normalized space its weights were trained on.
        with open(args.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded warm-start scaler from {args.scaler_path}")
    else:
        scaler = HeteroGraphScaler(
            whiten=not args.no_whiten,
            pca_components=args.pca_components,
        )
        scaler.fit(train_graphs_raw)

    train_graphs = [scaler.transform_graph(g) for g in train_graphs_raw]
    val_graphs = [scaler.transform_graph(g) for g in val_graphs_raw]
    test_graphs = [scaler.transform_graph(g) for g in test_graphs_raw]

    # Node feature width going into the first NNConv:
    #   profile -> 33 (8 base + 25 profile)
    #   cnn     -> 9 + latent_node_dim
    #   hybrid  -> 33 + latent_node_dim (profile features + CNN embedding)
    base_node_dim = train_graphs[0]["well"].x.shape[1]
    if args.node_encoder in ("cnn", "hybrid"):
        input_dim = base_node_dim + args.latent_node_dim
    else:
        input_dim = base_node_dim
    global_dim = train_graphs[0].global_attr.shape[1]

    print(f"Loaded {len(graphs_raw)} hetero graphs from {args.h5_path}")
    print(
        f"Target range: min={float(np.min(targets)):.4f}, max={float(np.max(targets)):.4f}"
    )
    print(f"Node feature dimension after scaling/whitening: {input_dim}")
    print(f"Global feature dimension: {global_dim}")
    print(
        f"Split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}"
    )

    if args.cache_to_gpu and torch.cuda.is_available():
        print(f"Caching entire dataset directly to GPU (cuda:{args.gpu}) VRAM...")
        device = torch.device(f"cuda:{args.gpu}")
        for split_subset in [train_graphs, val_graphs, test_graphs]:
            for g in split_subset:
                g.to(device)  # Moves standard PyG nodes/edges
                # Manually traverse our custom python wrapper protecting the 3D grid volumes
                for k, v in g.physics_context.d.items():
                    g.physics_context.d[k] = v.to(device, non_blocking=True)
        # Multiprocessing serialization crashes attempting to fork CUDA tensors via IPC
        args.num_workers = 0
        print("Set num_workers=0 to support native CUDA dataset persistence.")

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=not args.cache_to_gpu,
        generator=torch.Generator().manual_seed(run_seed),
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=not args.cache_to_gpu,
    )

    if args.checkpoint_path is not None:
        # PyTorch 2.6+ defaults weights_only=True; make pathlib safe to deserialize.
        import pathlib as _pl
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([_pl.PosixPath, _pl.WindowsPath])
        model = HeteroGNNRegressor.load_from_checkpoint(str(args.checkpoint_path))
        print(f"Warm-started model from {args.checkpoint_path}")
    else:
        model = HeteroGNNRegressor(
            input_dim=input_dim,
            global_dim=global_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pooling=args.pooling,
            residual=not args.no_residual,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            loss=args.loss,
            prediction_level=prediction_level,
            output_dim=output_dim,
            active_channels=[
                "PermX",
                "PermY",
                "PermZ",
                "Porosity",
                "Temperature0",
                "Pressure0",
                "valid_mask",
            ],
            latent_edge_dim=32,
            edge_encoder=args.edge_encoder,
            svd_weights_path=str(args.svd_weights_path) if args.svd_weights_path else None,
            node_encoder=args.node_encoder,
            latent_node_dim=args.latent_node_dim,
            node_pad=args.node_pad,
            node_z_extent=args.node_z_extent,
            edge_norm=args.edge_norm,
            edge_raw_means=args.edge_raw_means,
            node_aggr=args.node_aggr,
            cnn_activation=args.cnn_activation,
            head_no_norm=args.head_no_norm,
        )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_loss:.4f}",
        dirpath=Path(logger.log_dir) / "checkpoints",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )

    if args.gpu.lower() == "auto":
        devices = "auto"
    else:
        devices = [int(x.strip()) for x in args.gpu.split(",")]

    epochs_for_run = args.max_epochs
    if args.checkpoint_path is not None and args.max_epochs_finetune is not None:
        epochs_for_run = args.max_epochs_finetune
        print(f"Finetune cap: max_epochs={epochs_for_run}")

    trainer = L.Trainer(
        max_epochs=epochs_for_run,
        accelerator="gpu" if devices != "auto" else "auto",
        devices=devices,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip_val,
        deterministic=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_cb.best_model_path
    if not best_path:
        raise RuntimeError("No checkpoint was saved.")
    print(f"Best checkpoint: {best_path}")

    # Save scaler alongside checkpoint for inference reuse
    scaler_path = Path(best_path).parent / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler: {scaler_path}")

    # PyTorch 2.6 defaults weights_only=True which blocks pathlib.PosixPath deserialization
    import pathlib

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])

    best_model = HeteroGNNRegressor.load_from_checkpoint(best_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("mps")
    best_model = best_model.to(device)

    split_graphs = {
        "train": train_graphs,
        "val": val_graphs,
        "test": test_graphs,
    }

    target_labels = {
        "graph_energy_total": "Total Energy Production",
        "graph_energy_rate": "Energy Production Rate",
        "graph_discounted_net_revenue": "Discounted Net Revenue",
        "node_wept": "WEPT",
        "node_tp_final": "TP Profile Statistic",
    }
    target_label = target_labels.get(args.target, "Target")

    print("\nMetrics in original target units:")

    split_eval_data = {}
    metrics_report: dict[str, object] = {
        "target": args.target,
        "target_label": target_label,
        "prediction_level": prediction_level,
        "output_dim": output_dim,
        "splits": {},
    }
    metrics_log_lines: list[str] = ["Metrics in original target units:"]

    for split_name, graphs in split_graphs.items():
        y_true, y_pred, case_ids = evaluate_split(
            best_model, graphs, scaler, args.batch_size, device
        )
        split_eval_data[split_name] = (y_true, y_pred)

        metrics = compute_metrics(y_true, y_pred)
        split_line = (
            f"  {split_name:<5} | MAE={metrics['mae']:>10.1f} | "
            f"MedAE={metrics['medae']:>10.1f} | RMSE={metrics['rmse']:>10.1f} | "
            f"MAPE={metrics['mape']:>5.1f}% | R2={metrics['r2']:>6.4f}"
        )
        print(split_line)
        metrics_log_lines.append(split_line)

        split_report: dict[str, object] = {
            "case_count": len(set(case_ids)),
            "sample_count": int(y_true.shape[0]),
            "metrics": metrics,
        }

        # Per-statistic breakdown for multi-output targets
        if output_dim > 1 and y_true.shape[1] == output_dim:
            stat_names = {
                TP_PROFILE_STATS: [
                    "mean_T",
                    "min_T",
                    "max_T",
                    "mean_P",
                    "min_P",
                    "max_P",
                ],
            }
            names = stat_names.get(output_dim, [f"dim_{j}" for j in range(output_dim)])
            per_output_report: dict[str, dict[str, float]] = {}
            for j, name in enumerate(names):
                m_j = compute_metrics(y_true[:, j : j + 1], y_pred[:, j : j + 1])
                dim_line = (
                    f"    {name:>8s}: MAE={m_j['mae']:>8.2f} | "
                    f"MAPE={m_j['mape']:>5.1f}% | R2={m_j['r2']:>7.4f}"
                )
                print(dim_line)
                metrics_log_lines.append(dim_line)
                per_output_report[name] = m_j
            split_report["per_output_metrics"] = per_output_report

        metrics_report["splits"][split_name] = split_report

        save_predictions_csv(
            plots_dir / f"{split_name}_predictions.csv",
            split_name,
            case_ids,
            y_true,
            y_pred,
        )
        save_extreme_error_plots(
            plots_dir, split_name, case_ids, y_true, y_pred, top_k=args.extreme_k
        )
        save_extreme_error_graph_plots(
            plots_dir, split_name, graphs, case_ids, y_true, y_pred
        )

    save_error_scatter_plots(plots_dir, split_eval_data, target_label=target_label)

    metrics_txt_path = plots_dir / "metrics_summary.txt"
    metrics_txt_path.write_text("\n".join(metrics_log_lines) + "\n")
    print(f"Saved metrics text summary: {metrics_txt_path}")

    metrics_json_path = plots_dir / "metrics_summary.json"
    with metrics_json_path.open("w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Saved metrics JSON summary: {metrics_json_path}")

    save_loss_curve_plot(
        Path(logger.log_dir) / "metrics.csv", plots_dir / "loss_over_time.png"
    )


if __name__ == "__main__":
    main()
