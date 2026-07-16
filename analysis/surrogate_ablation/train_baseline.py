#!/usr/bin/env python3
"""Train a baseline model (MLP / global 3D CNN) for the architecture ablation.

Mirrors train.py's data pipeline (identical load_hetero_graphs call,
geology-stratified split, HeteroGraphScaler target whitening) and its output
layout (<output-root>/baseline/run_NN/{checkpoints/best-*.ckpt, checkpoints/
scaler.pkl, plots/metrics_summary.json, plots/{split}_predictions.csv}) so the
matrix runner can treat GNN and baseline runs uniformly.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from baselines import (  # noqa: E402
    GlobalCNN3D,
    MLPDataset,
    N_GEOLOGIES,
    VolumeDataset,
    WellMLPBaseline,
    build_case_arrays,
    grid_sample_unit_check,
    load_split_scale,
)
from geothermal.evaluation import compute_metrics, save_predictions_csv  # noqa: E402

TARGET_LABELS = {
    "graph_energy_total": "Total Energy Production",
    "graph_energy_rate": "Energy Production Rate",
    "graph_discounted_net_revenue": "Discounted Net Revenue",
    "node_wept_final": "WEPT (final year)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", type=Path, required=True)
    p.add_argument("--target", required=True, choices=list(TARGET_LABELS))
    p.add_argument("--model", required=True, choices=["mlp", "global_cnn"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--output-root", type=Path, default=Path("lightning_logs"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--max-epochs", type=int, default=180)
    p.add_argument("--early-stop-patience", type=int, default=60)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--cache-to-gpu", action="store_true")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--train-subsample", type=int, default=None)
    p.add_argument("--holdout-geologies", type=str, default=None)
    p.add_argument("--require-geology-map", action="store_true",
                   help="No-op (map is always required here); accepted for "
                        "runner argv uniformity.")
    # arch knobs
    p.add_argument("--hidden-dims", default="512,512,256", help="MLP hidden sizes")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--widths", default="16,32,64,128", help="global_cnn conv widths")
    p.add_argument("--marker-sigma", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    run_seed = args.seed + args.run_id
    L.seed_everything(run_seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grid_sample_unit_check()

    holdout = ([int(x) for x in args.holdout_geologies.split(",")]
               if args.holdout_geologies else None)
    graphs, splits, scaler, geo_idx = load_split_scale(
        args.h5_path, args.target, split_seed,
        args.val_fraction, args.test_fraction, args.max_cases,
        holdout_geologies=holdout,
        train_subsample=args.train_subsample,
        subsample_seed=args.seed + args.run_id,
    )
    prediction_level = "node" if args.target.startswith("node_") else "graph"
    cases = [
        build_case_arrays(g, int(geo_idx[i]), scaler.target_scaler)
        for i, g in enumerate(graphs)
    ]
    n_wells_all = {c["feats"].shape[0] for c in cases}
    assert len(n_wells_all) == 1, f"variable well counts: {sorted(n_wells_all)}"
    n_wells = n_wells_all.pop()

    logger = CSVLogger(save_dir=str(args.output_root), name="baseline",
                       version=f"run_{args.run_id:02d}")
    log_dir = Path(logger.log_dir)
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ----- model + datasets -----
    if args.model == "mlp":
        feat_scaler = StandardScaler()
        train_wells = np.concatenate([cases[i]["feats"] for i in splits["train"]])
        feat_scaler.fit(train_wells)

        def flat_x(c):
            return np.concatenate(
                [feat_scaler.transform(c["feats"]).reshape(-1), c["geo_onehot"]]
            ).astype(np.float32)

        def make_ds(idx):
            X = torch.tensor(np.stack([flat_x(cases[i]) for i in idx]))
            if prediction_level == "node":
                Y = torch.tensor(np.stack([cases[i]["y_scaled"][:, 0] for i in idx]))
                M = torch.tensor(np.stack([cases[i]["prod_mask"] for i in idx]))
            else:
                Y = torch.tensor(np.stack([cases[i]["y_scaled"][0] for i in idx]))
                M = torch.zeros(len(idx), 1, dtype=torch.bool)
            if args.cache_to_gpu and device.type == "cuda":
                X, Y, M = X.to(device), Y.to(device), M.to(device)
            return MLPDataset(X, Y, M, [cases[i]["case_id"] for i in idx])

        lr = args.learning_rate or 1e-3
        model = WellMLPBaseline(
            n_wells=n_wells, prediction_level=prediction_level,
            hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",")),
            dropout=args.dropout, learning_rate=lr,
        )
    else:
        cache_dev = device if (args.cache_to_gpu and device.type == "cuda") else None

        def make_ds(idx):
            return VolumeDataset([graphs[i] for i in idx],
                                 [cases[i] for i in idx], device=cache_dev)

        lr = args.learning_rate or 3e-4
        model = GlobalCNN3D(
            prediction_level=prediction_level,
            widths=tuple(int(x) for x in args.widths.split(",")),
            marker_sigma=args.marker_sigma, learning_rate=lr,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[baseline] model={args.model} params={n_params:,} "
          f"prediction_level={prediction_level} n_wells={n_wells}")

    datasets = {name: make_ds(idx) for name, idx in splits.items()}
    gen = torch.Generator().manual_seed(run_seed)
    loaders = {
        name: DataLoader(
            ds, batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=0, generator=gen if name == "train" else None,
        )
        for name, ds in datasets.items()
    }

    ckpt_cb = ModelCheckpoint(
        dirpath=log_dir / "checkpoints", monitor="val_loss", mode="min",
        save_top_k=1, filename="best-{epoch:03d}-{val_loss:.4f}",
    )
    early = EarlyStopping(monitor="val_loss", patience=args.early_stop_patience,
                          min_delta=1e-5)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=[int(args.gpu)] if device.type == "cuda" else 1,
        callbacks=[ckpt_cb, early], logger=logger, log_every_n_steps=1,
        gradient_clip_val=1.0, deterministic=False, enable_progress_bar=False,
    )
    trainer.fit(model, loaders["train"], loaders["val"])

    with open(log_dir / "checkpoints" / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ----- evaluation with the best checkpoint -----
    cls = WellMLPBaseline if args.model == "mlp" else GlobalCNN3D
    best = cls.load_from_checkpoint(ckpt_cb.best_model_path, map_location=device)
    best.eval()

    metrics_report = {
        "target": args.target,
        "target_label": TARGET_LABELS[args.target],
        "prediction_level": prediction_level,
        "output_dim": 1,
        "model": args.model,
        "params": n_params,
        "splits": {},
    }
    print("\nMetrics in original target units:")
    for split_name, ds in datasets.items():
        preds, trues, cids = [], [], []
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                batch = [t.to(device) for t in batch]
                if args.model == "mlp":
                    x, y, m = batch
                    pred = best(x)
                else:
                    vol7, pos, inj, y, m = batch
                    pred = best(vol7, pos, inj)
                start = bi * args.batch_size
                if prediction_level == "node":
                    for r in range(pred.shape[0]):
                        pm = m[r].cpu().numpy()
                        preds.append(pred[r].cpu().numpy()[pm, None])
                        trues.append(y[r].cpu().numpy()[pm, None])
                        cids.extend([ds.case_ids[start + r]] * int(pm.sum()))
                else:
                    preds.append(pred.reshape(-1, 1).cpu().numpy())
                    trues.append(y.reshape(-1, 1).cpu().numpy())
                    cids.extend(ds.case_ids[start:start + pred.shape[0]])
        y_pred = scaler.inverse_targets(np.concatenate(preds))
        y_true = scaler.inverse_targets(np.concatenate(trues))
        metrics = compute_metrics(y_true, y_pred)
        print(f"  {split_name:<5} | MAE={metrics['mae']:>10.1f} | "
              f"MedAE={metrics['medae']:>10.1f} | RMSE={metrics['rmse']:>10.1f} | "
              f"MAPE={metrics['mape']:>5.1f}% | R2={metrics['r2']:>6.4f}")
        metrics_report["splits"][split_name] = {
            "case_count": len(set(cids)),
            "sample_count": int(y_true.shape[0]),
            "metrics": metrics,
        }
        save_predictions_csv(plots_dir / f"{split_name}_predictions.csv",
                             split_name, cids, y_true, y_pred)

    with open(plots_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"Saved metrics JSON summary: {plots_dir / 'metrics_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
