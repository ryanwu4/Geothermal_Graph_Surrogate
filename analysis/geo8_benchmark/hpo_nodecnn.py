"""Hyperparameter optimization for the node-CNN variant.

Search space targets overfit-reduction: dropout, weight_decay, learning_rate.
Each trial trains train.py as a subprocess with --node-encoder cnn and reads
back the best val_loss from the metrics CSV.

Outputs:
  hpo_mse_nodecnn/trials.csv            — per-trial hyperparams + best val_loss
  hpo_mse_nodecnn/best.json             — best trial config + score
  hpo_mse_nodecnn/trial_NN/             — train.py output dirs per trial
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
import pandas as pd

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent  # Geothermal_Graph_Surrogate
HPO_ROOT = THIS / "runs" / "hpo_mse_nodecnn"


def get_best_val_loss(trial_root: Path) -> float:
    metrics_csv = trial_root / "geothermal_hetero_gnn" / "run_00" / "metrics.csv"
    if not metrics_csv.exists():
        return float("inf")
    df = pd.read_csv(metrics_csv)
    if "val_loss" not in df.columns:
        return float("inf")
    vl = df["val_loss"].dropna()
    if len(vl) == 0:
        return float("inf")
    return float(vl.min())


def objective(trial: optuna.Trial) -> float:
    dropout       = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay  = trial.suggest_float("weight_decay", 1e-3, 3e-1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [8, 16, 32])
    node_pad      = trial.suggest_categorical("node_pad", [3, 5])
    latent_node_dim = trial.suggest_categorical("latent_node_dim", [16, 32, 64])

    trial_dir = HPO_ROOT / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"

    cmd = [
        sys.executable, str(REPO / "train.py"),
        "--h5-path", "analysis/geo8_benchmark/benchmark.h5",
        "--target", "graph_discounted_net_revenue",
        "--seed", "42",
        "--split-seed", "42",
        "--run-id", "0",
        "--output-root", str(trial_dir),
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--learning-rate", f"{learning_rate:.6f}",
        "--max-epochs", "120",
        "--early-stop-patience", "20",
        "--edge-encoder", "cnn",
        "--stratified-split",
        "--cache-to-gpu",
        "--loss", "mse",
        "--node-encoder", "cnn",
        "--node-pad", str(node_pad),
        "--latent-node-dim", str(latent_node_dim),
        "--dropout", f"{dropout:.4f}",
        "--weight-decay", f"{weight_decay:.6f}",
    ]
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    if proc.returncode != 0:
        print(f"Trial {trial.number}: train.py failed (rc={proc.returncode})")
        return float("inf")

    val_loss = get_best_val_loss(trial_dir)
    print(f"Trial {trial.number}: dropout={dropout:.3f} wd={weight_decay:.2e} lr={learning_rate:.2e} "
          f"bs={batch_size} pad={node_pad} latent={latent_node_dim} -> val_loss={val_loss:.4f}")
    return val_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    HPO_ROOT.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="nodecnn_overfit",
    )

    # First trial: replicate the original (overfitting) config as a baseline
    study.enqueue_trial({
        "dropout": 0.0,
        "weight_decay": 1e-2,
        "learning_rate": 3e-4,
        "batch_size": 16,
        "node_pad": 3,
        "latent_node_dim": 32,
    })

    study.optimize(objective, n_trials=args.n_trials)

    print()
    print(f"Best trial #{study.best_trial.number}: val_loss = {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "val_loss": t.value, **t.params}
            trials.append(row)
    df = pd.DataFrame(trials).sort_values("val_loss")
    df.to_csv(HPO_ROOT / "trials.csv", index=False)
    (HPO_ROOT / "best.json").write_text(json.dumps({
        "best_trial": study.best_trial.number,
        "best_val_loss": study.best_value,
        "best_params": study.best_params,
    }, indent=2))
    print(f"\nTop 5 trials:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
