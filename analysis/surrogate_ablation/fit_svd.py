#!/usr/bin/env python3
"""Fit an SVD/PCA basis for PhysicsSlabSVD on TRAIN-SPLIT slabs only.

Replaces scripts/precompute_svd.py for the ablation study: that script fits on
every case (test leakage) and concatenates every edge slab (unbounded RAM).
This one reuses the exact split recipe shared by train.py/train_baseline.py
(valid for all targets — stratification is geology-only), subsamples cases and
caps total slabs, and writes the same {components, mean, std} format that
PhysicsSlabSVD loads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from baselines import load_split_scale  # noqa: E402
from geothermal.model import EDGE_TYPES  # noqa: E402
from geothermal.physics_slab import PhysicsSlabExtractor  # noqa: E402

ACTIVE_CHANNELS = [
    "PermX", "PermY", "PermZ", "Porosity", "Temperature0", "Pressure0", "valid_mask",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--max-cases-fit", type=int, default=300)
    p.add_argument("--max-slabs", type=int, default=12000)
    p.add_argument("--max-cases", type=int, default=None,
                   help="Dataset truncation (must match the jobs' --max-cases "
                        "so the fitted split is the same)")
    p.add_argument("--train-subsample", type=int, default=None,
                   help="Match the jobs' train subsample so the basis is fit "
                        "on exactly the cases available to training")
    p.add_argument("--subsample-seed", type=int, default=None)
    p.add_argument("--holdout-geologies", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    graphs, splits, _, _ = load_split_scale(
        args.h5_path, "graph_energy_total", args.split_seed,
        args.val_fraction, args.test_fraction, args.max_cases,
        holdout_geologies=([int(x) for x in args.holdout_geologies.split(",")]
                           if args.holdout_geologies else None),
        train_subsample=args.train_subsample,
        subsample_seed=args.subsample_seed,
    )
    rng = np.random.default_rng(args.seed)
    train_idx = np.array(splits["train"])
    fit_idx = rng.permutation(train_idx)[: args.max_cases_fit]
    print(f"[fit_svd] fitting on {len(fit_idx)} train-split cases "
          f"(of {len(train_idx)} train / {len(graphs)} total)")

    extractor = PhysicsSlabExtractor(active_channels=ACTIVE_CHANNELS)
    device = torch.device(args.device)
    slab_parts: list[torch.Tensor] = []
    n_slabs = 0
    for gi in fit_idx:
        g = graphs[int(gi)]
        pos = g["well"].pos_xyz.to(device)
        phys = {k: v.to(device) for k, v in g.physics_context.d.items()}
        full_shape = g.physics_context.full_shape
        for et in EDGE_TYPES:
            ei = g[et].edge_index
            if ei.numel() == 0:
                continue
            ca, cb = pos[ei[0]], pos[ei[1]]
            expanded = {
                k: v.unsqueeze(0).expand(ca.shape[0], -1, -1, -1) for k, v in phys.items()
            }
            with torch.no_grad():
                slabs = extractor(expanded, ca, cb, full_shape)
            slab_parts.append(slabs.reshape(slabs.shape[0], -1).cpu())
            n_slabs += slabs.shape[0]
        if n_slabs >= args.max_slabs:
            print(f"[fit_svd] slab cap reached ({n_slabs} >= {args.max_slabs})")
            break

    X = torch.cat(slab_parts, dim=0)[: args.max_slabs]
    print(f"[fit_svd] slab matrix: {tuple(X.shape)} "
          f"({X.numel() * 4 / 1e9:.1f} GB float32)")

    mean = X.mean(dim=0)
    std = X.std(dim=0)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    Xs = (X - mean) / std
    _, _, V = torch.pca_lowrank(Xs, q=args.k, center=False)
    state = {"components": V.t().contiguous(), "mean": mean, "std": std}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output_path)
    print(f"[fit_svd] saved components {tuple(state['components'].shape)} "
          f"to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
