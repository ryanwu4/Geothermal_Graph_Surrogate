"""Run 8 profile-feature ablations in parallel (4 concurrent, 2 per GPU), then evaluate each.

For each ablation:
  1. Spawn train.py with --ablate <name> on a designated GPU.
  2. Once training finishes, spawn evaluate_benchmark.py with the same --ablate flag.
  3. Collect per-cohort metrics into ablations/summary.csv.

Reference (no ablation) is the existing analysis/geo8_benchmark/training_mse/ run.
Hardware: 2 A100 80GB GPUs; each training fits at ~half-GPU, so 2 trainings per GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent.parent  # Geothermal_Graph_Surrogate
ABL_ROOT = THIS / "runs" / "ablations"

# Ablation -> GPU assignment. Interleaved so that with ThreadPoolExecutor(max_workers=4)
# the first wave (first 4 to be scheduled) hits GPUs 0,1,0,1 (2 per GPU).
ABLATIONS: list[tuple[str, str]] = [
    ("remove_perm_profile",   "0"),
    ("remove_poro_profile",   "1"),
    ("remove_thermo_profile", "0"),
    ("keep_only_means",       "1"),
    ("keep_only_extrema",     "0"),
    ("keep_only_std",         "1"),
    ("keep_only_n_layers",    "0"),
    ("remove_all_profile",    "1"),
]


def train_one(ablation: str, gpu: str) -> tuple[str, int]:
    train_root = ABL_ROOT / ablation
    train_root.mkdir(parents=True, exist_ok=True)
    log_path = train_root / "train.log"
    # Isolate the subprocess to one physical GPU via CUDA_VISIBLE_DEVICES.
    # train.py's --gpu flag then sees a single device (re-indexed to 0).
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        sys.executable, str(REPO / "train.py"),
        "--h5-path", "analysis/geo8_benchmark/benchmark.h5",
        "--target", "graph_discounted_net_revenue",
        "--seed", "42", "--split-seed", "42",
        "--run-id", "0",
        "--output-root", str(train_root),
        "--gpu", "0",   # single visible GPU (post-CUDA_VISIBLE_DEVICES masking)
        "--batch-size", "16",
        "--learning-rate", "3e-4",
        "--max-epochs", "180",
        "--early-stop-patience", "30",
        "--edge-encoder", "cnn",
        "--stratified-split",
        "--cache-to-gpu",
        "--loss", "mse",
        "--node-encoder", "profile",
        "--ablate", ablation,
    ]
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
    return ablation, proc.returncode


def eval_one(ablation: str) -> tuple[str, int]:
    train_root = ABL_ROOT / ablation
    eval_dir = ABL_ROOT / f"eval_{ablation}"
    log_path = eval_dir / "eval.log"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "analysis/geo8_benchmark/evaluate_benchmark.py",
        "--training-root", str(train_root),
        "--out-dir", str(eval_dir),
        "--node-encoder", "profile",
        "--ablate", ablation,
    ]
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    return ablation, proc.returncode


def collate() -> None:
    rows = []
    # Reference row from the existing MSE baseline.
    ref_metrics = json.load(open(THIS / "runs" / "eval_mse" / "metrics.json"))
    def _extract(name: str, m: dict) -> dict:
        return {
            "ablation": name,
            "aggregate_mape": m["aggregate"]["mape"],
            "geo8_mape":      m["geo8"]["mape"],
            "non_geo8_mape":  m["non_geo8"]["mape"],
            "aggregate_r2":   m["aggregate"]["r2"],
            "geo8_r2":        m["geo8"]["r2"],
            "non_geo8_r2":    m["non_geo8"]["r2"],
            "geo8_test_mape": m["geo8_by_split"]["test"]["mape"],
            "geo8_test_r2":   m["geo8_by_split"]["test"]["r2"],
            "all_test_mape":  m["by_split"]["test"]["mape"],
            "all_test_r2":    m["by_split"]["test"]["r2"],
        }
    rows.append(_extract("reference_all_features", ref_metrics))
    for ablation, _ in ABLATIONS:
        mpath = ABL_ROOT / f"eval_{ablation}/metrics.json"
        if not mpath.exists():
            print(f"  WARNING: no metrics for {ablation}")
            continue
        rows.append(_extract(ablation, json.load(open(mpath))))

    out = ABL_ROOT / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")

    # Pretty print
    print()
    print(f"{'ablation':<26} {'all_MAPE':>9} {'!g8_MAPE':>9} {'g8_MAPE':>9} | {'all_R2':>7} {'!g8_R2':>7} {'g8_R2':>7} | {'g8t_MAPE':>9} {'g8t_R2':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r['ablation']:<26} {r['aggregate_mape']:>9.2f} {r['non_geo8_mape']:>9.2f} {r['geo8_mape']:>9.2f}"
              f" | {r['aggregate_r2']:>+7.3f} {r['non_geo8_r2']:>+7.3f} {r['geo8_r2']:>+7.3f}"
              f" | {r['geo8_test_mape']:>9.2f} {r['geo8_test_r2']:>+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="Skip training, just collate existing runs")
    ap.add_argument("--skip-eval", action="store_true", help="Skip evaluation, just collate existing")
    args = ap.parse_args()

    ABL_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        print("=== TRAINING (4 concurrent: 2 per GPU) ===")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(train_one, ab, gpu): ab for ab, gpu in ABLATIONS}
            for f in as_completed(futs):
                ab, rc = f.result()
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [train done] {ab}: {status}")

    if not args.skip_eval:
        print("=== EVALUATING (4 concurrent) ===")
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(eval_one, ab): ab for ab, _ in ABLATIONS}
            for f in as_completed(futs):
                ab, rc = f.result()
                status = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [eval done] {ab}: {status}")

    print("=== COLLATING ===")
    collate()


if __name__ == "__main__":
    main()
