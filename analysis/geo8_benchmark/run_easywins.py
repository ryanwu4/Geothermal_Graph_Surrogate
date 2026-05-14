"""Run the 4 Tier-B easy-win A/B tests in parallel (2 per GPU), then evaluate each.

Each test layers ONE additional change on top of the Exp 1 baseline (v3 node CNN
+ edge BN→GN + edge raw means + MSE loss + full-Z slab). The 4 tests:
  - B1: --enrich-global-attr
  - B2: --node-aggr sum    (LR halved as a safety against larger gradients)
  - B3: --cnn-activation gelu
  - B4: --head-no-norm

Output dirs: analysis/geo8_benchmark/easywins/<B_name>/{training,eval_}/
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent
ROOT = THIS / "runs" / "easywins"

# (name, extra_train_args, gpu) — interleave GPU 0/1 so first wave = 2 per GPU.
TESTS: list[tuple[str, list[str], str]] = [
    ("B1_global_attr",  ["--enrich-global-attr"],                       "0"),
    ("B2_sum_aggr",     ["--node-aggr", "sum", "--learning-rate", "1.5e-4"], "1"),
    ("B3_gelu",         ["--cnn-activation", "gelu"],                   "0"),
    ("B4_no_head_norm", ["--head-no-norm"],                             "1"),
]

# Baseline args (= Exp 1: edge GN + raw means, MSE, v3 node CNN, full-Z slab).
BASE_ARGS = [
    "--h5-path", "analysis/geo8_benchmark/benchmark.h5",
    "--target", "graph_discounted_net_revenue",
    "--seed", "42", "--split-seed", "42",
    "--run-id", "0",
    "--batch-size", "16",
    "--max-epochs", "180",
    "--early-stop-patience", "30",
    "--edge-encoder", "cnn",
    "--edge-norm", "groupnorm",
    "--edge-raw-means",
    "--node-encoder", "cnn",
    "--node-pad", "3",
    "--node-z-extent", "full",
    "--latent-node-dim", "32",
    "--stratified-split",
    "--cache-to-gpu",
    "--loss", "mse",
]
# LR default (only used if test doesn't override).
DEFAULT_LR = ["--learning-rate", "3e-4"]


def train_one(name: str, extra: list[str], gpu: str) -> tuple[str, int]:
    train_root = ROOT / name / "training"
    train_root.mkdir(parents=True, exist_ok=True)
    log_path = train_root / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # If --learning-rate is in extra, drop the default LR from BASE_ARGS.
    has_lr = "--learning-rate" in extra
    cmd = [sys.executable, str(REPO / "train.py")] + BASE_ARGS + [
        "--output-root", str(train_root),
        "--gpu", "0",  # post-CUDA_VISIBLE_DEVICES masking
    ]
    if not has_lr:
        cmd += DEFAULT_LR
    cmd += extra
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
    return name, proc.returncode


def eval_one(name: str) -> tuple[str, int]:
    train_root = ROOT / name / "training"
    eval_dir = ROOT / name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "eval.log"
    # Some tests need flags forwarded to eval too (--enrich-global-attr only needs
    # to affect data loading; --node-aggr / --cnn-activation / --head-no-norm are
    # baked into the checkpoint's saved_hyperparameters).
    extra_eval = []
    if name == "B1_global_attr":
        extra_eval += ["--enrich-global-attr"]
    cmd = [
        sys.executable, "analysis/geo8_benchmark/evaluate_benchmark.py",
        "--training-root", str(train_root),
        "--out-dir", str(eval_dir),
        "--node-encoder", "cnn",
    ] + extra_eval
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    return name, proc.returncode


def collate() -> None:
    rows = []
    ref_metrics = json.load(open(THIS / "runs" / "eval_mse_edgegn" / "metrics.json"))

    def _extract(name: str, m: dict) -> dict:
        return {
            "test": name,
            "aggregate_mape": m["aggregate"]["mape"],
            "geo8_mape":      m["geo8"]["mape"],
            "non_geo8_mape":  m["non_geo8"]["mape"],
            "aggregate_r2":   m["aggregate"]["r2"],
            "geo8_r2":        m["geo8"]["r2"],
            "non_geo8_r2":    m["non_geo8"]["r2"],
            "all_test_mape":  m["by_split"]["test"]["mape"],
            "geo8_test_mape": m["geo8_by_split"]["test"]["mape"],
            "geo8_test_r2":   m["geo8_by_split"]["test"]["r2"],
        }

    rows.append(_extract("baseline_Exp1_edgegn", ref_metrics))
    for name, _, _ in TESTS:
        mpath = ROOT / name / "eval" / "metrics.json"
        if not mpath.exists():
            print(f"  WARNING: no metrics for {name}")
            continue
        rows.append(_extract(name, json.load(open(mpath))))

    out_csv = ROOT / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    print()
    print(f"{'test':<22} {'all_MAPE':>9} {'!g8_MAPE':>9} {'g8_MAPE':>9} | {'all_R2':>7} {'!g8_R2':>7} {'g8_R2':>7} | {'g8t_MAPE':>9} {'g8t_R2':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r['test']:<22} {r['aggregate_mape']:>9.2f} {r['non_geo8_mape']:>9.2f} {r['geo8_mape']:>9.2f}"
              f" | {r['aggregate_r2']:>+7.3f} {r['non_geo8_r2']:>+7.3f} {r['geo8_r2']:>+7.3f}"
              f" | {r['geo8_test_mape']:>9.2f} {r['geo8_test_r2']:>+7.3f}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    print("=== TRAINING (4 concurrent: 2 per GPU) ===")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(train_one, name, extra, gpu): name for name, extra, gpu in TESTS}
        for f in as_completed(futs):
            name, rc = f.result()
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"  [train done] {name}: {status}")

    print("=== EVALUATING (4 concurrent) ===")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(eval_one, name): name for name, _, _ in TESTS}
        for f in as_completed(futs):
            name, rc = f.result()
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"  [eval done] {name}: {status}")

    print("=== COLLATING ===")
    collate()


if __name__ == "__main__":
    main()
