#!/usr/bin/env python3
"""Run inference with a trained GNN to predict total energy production.

Usage:
    python infer.py \\
        --h5-path data/compiled.h5 \\
        --checkpoint lightning_logs/geothermal_hetero_gnn/version_0/checkpoints/best-*.ckpt \\
        --scaler-path lightning_logs/geothermal_hetero_gnn/version_0/checkpoints/scaler.pkl \\
        --output predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from geothermal.data import HeteroGraphScaler, load_hetero_graphs
from geothermal.model import HeteroGNNRegressor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer total energy production from compiled H5 data using a trained GNN."
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        required=True,
        help="Path to the compiled H5 dataset for inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        required=True,
        help="Path to the fitted scaler pickle (scaler.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV path for predictions. Default: predictions.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference. Default: 32",
    )
    parser.add_argument(
        "--target",
        choices=[
            "node_wept",
            "node_tp_final",
            "graph_energy_total",
            "graph_energy_rate",
        ],
        default="graph_energy_total",
        help="Prediction target (must match what the model was trained on).",
    )
    args = parser.parse_args()

    # --- Load scaler ---
    print(f"Loading scaler from {args.scaler_path}")
    with open(args.scaler_path, "rb") as f:
        scaler: HeteroGraphScaler = pickle.load(f)

    # --- Load model ---
    print(f"Loading model from {args.checkpoint}")
    model = HeteroGNNRegressor.load_from_checkpoint(str(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("mps")
    model = model.to(device)
    model.eval()

    # --- Load and transform data ---
    print(f"Loading graphs from {args.h5_path}")
    graphs_raw, targets = load_hetero_graphs(args.h5_path, target=args.target)
    print(f"Loaded {len(graphs_raw)} graphs.")

    graphs = [scaler.transform_graph(g) for g in graphs_raw]

    # --- Inference ---
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    all_preds_scaled: list[np.ndarray] = []
    all_case_ids: list[str] = []

    is_node_level = model.prediction_level == "node"
    filter_ext = getattr(graphs[0], "filter_extractors", True) if graphs else False
    should_filter = is_node_level and filter_ext

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            if should_filter:
                is_inj = batch["well"].is_injector.detach().cpu().numpy()
                ext_mask = is_inj < 0.5
                all_preds_scaled.append(pred.detach().cpu().numpy()[ext_mask])
                for i in range(batch.num_graphs):
                    start = batch["well"].ptr[i].item()
                    end = batch["well"].ptr[i + 1].item()
                    n_ext = np.sum(ext_mask[start:end])
                    all_case_ids.extend([batch.case_id[i]] * int(n_ext))
            elif is_node_level:
                all_preds_scaled.append(pred.detach().cpu().numpy())
                for i in range(batch.num_graphs):
                    start = batch["well"].ptr[i].item()
                    end = batch["well"].ptr[i + 1].item()
                    n_nodes = end - start
                    all_case_ids.extend([batch.case_id[i]] * n_nodes)
            else:
                all_preds_scaled.append(pred.detach().cpu().numpy())
                all_case_ids.extend(list(batch.case_id))

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    preds = scaler.inverse_targets(preds_scaled)

    # Also inverse-transform true targets for comparison
    true_targets = targets  # already in original scale from load_hetero_graphs

    # --- Write output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        if is_node_level:
            header = ["case_id", "node_index"]
            for j in range(preds.shape[1]):
                header.append(f"predicted_dim_{j}")
            writer.writerow(header)
            for i, cid in enumerate(all_case_ids):
                row = [cid, i] + [float(v) for v in preds[i]]
                writer.writerow(row)
        else:
            writer.writerow(
                ["case_id", "predicted_total_energy", "actual_total_energy"]
            )
            for i, cid in enumerate(all_case_ids):
                pred_val = float(preds[i].flat[0])
                actual_val = float(true_targets[i]) if i < len(true_targets) else ""
                writer.writerow([cid, pred_val, actual_val])

    print(f"Wrote {len(all_case_ids)} predictions to {args.output}")


if __name__ == "__main__":
    main()
