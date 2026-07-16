#!/usr/bin/env python3
"""Cross-dataset evaluation of stage-1 checkpoints (no retraining).

Every (variant, target, train_dataset, seed) checkpoint is evaluated on the
TEST split of every dataset (including its own — the diagonal reproduces the
within-dataset numbers as a pipeline sanity check). Valid because all three
datasets share byte-identical physics normalization; each checkpoint's own
scaler (feature/target whitening) travels with it.

Constraints handled:
- The MLP's input width is bound to the well count → transfers only between
  same-well-count datasets (hard_fulldepth <-> highperf_cma).
- Its per-well feature scaler was not persisted by train_baseline.py; it is
  refit deterministically from the TRAIN dataset's train split via a light
  H5 reader (no physics tensors), reproducing training exactly.

Usage:
    python cross_eval.py --stage-root runs/stage1 [--gpu 0] [--seed 42]
Writes <stage-root>/cross_eval.csv.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from baselines import (  # noqa: E402
    GlobalCNN3D,
    N_GEOLOGIES,
    VolumeDataset,
    WellMLPBaseline,
    build_case_arrays,
    canonical_well_order,
)
from geothermal.data import (  # noqa: E402
    load_hetero_graphs,
    peek_data_kwargs_from_checkpoint,
    split_indices_stratified,
)
from geothermal.evaluation import evaluate_split  # noqa: E402
from geothermal.model import HeteroGNNRegressor  # noqa: E402

DATASETS = {
    "simple_2pair": "ablation_datasets/simple_2pair/seed_compiled.h5",
    "hard_fulldepth": "ablation_datasets/hard_fulldepth/seed_compiled.h5",
    "highperf_cma": "ablation_datasets/highperf_cma/compiled.h5",
}
TARGETS = ["graph_energy_total", "graph_discounted_net_revenue", "node_wept_final"]
GNN_VARIANTS = ["gnn_default", "gnn_dist_edge", "gnn_type_only_nodes", "gnn_svd_edge"]
BASELINES = ["mlp", "global_cnn"]
# Split parameters are derived from a stage-1 job.json at startup (audit F8a) —
# these are only the fallbacks if none is found.
SPLIT_SEED = 42
FRACTIONS = (0.15, 0.15)


def derive_split_params(stage: Path) -> tuple[int, tuple[float, float]]:
    """Read --split-seed/--val-fraction/--test-fraction from any job.json."""
    for jj in sorted(stage.glob("*/*/*/seed*/job.json")):
        argv = json.loads(jj.read_text()).get("cmd", [])
        def grab(flag, default):
            return float(argv[argv.index(flag) + 1]) if flag in argv else default
        return (int(grab("--split-seed", SPLIT_SEED)),
                (grab("--val-fraction", FRACTIONS[0]),
                 grab("--test-fraction", FRACTIONS[1])))
    return SPLIT_SEED, FRACTIONS


def light_case_table(h5_path: Path, target: str):
    """case_ids, target scalars, wells arrays, geology idx — no physics tensors.

    MUST mirror load_hetero_graphs' semantics exactly (audit F3): skip 0-well
    cases, producer mask = NOT(inj_rate > 0) (zero-rate wells count as
    producers). The diagonal invariant in main() guards this equivalence.
    """
    gmap = json.loads((h5_path.parent / "case_geology_map.json").read_text())
    ids, targets, wells_list, wept = [], [], [], []
    with h5py.File(h5_path, "r") as f:
        for cid in sorted(f.keys()):
            g = f[cid]
            w = g["wells"][:]
            if len(w) == 0:  # mirror data.py's skipped_empty
                continue
            if target == "graph_energy_total":
                t = float(g["field_energy_production_total"][-1])
            elif target == "graph_discounted_net_revenue":
                t = float(g["field_discounted_net_revenue"][()])
            else:  # node_wept_final: stratification target = producer mean
                ww = g["well_wept"][:, -1]
                ext = ~(w["inj_rate"] > 0)  # mirror data.py is_injector < 0.5
                t = float(ww[ext].mean()) if ext.any() else 0.0
                wept.append(ww)
            ids.append(cid)
            targets.append(t)
            wells_list.append(w)
    geo = np.array([int(gmap[c]["geology_index"]) for c in ids], dtype=np.int64)
    return ids, np.array(targets, dtype=np.float32), wells_list, geo, wept


_SPLIT_PARAMS: dict = {"seed": SPLIT_SEED, "fractions": FRACTIONS}


def split_for(targets, geo):
    return split_indices_stratified(
        targets=targets,
        val_fraction=_SPLIT_PARAMS["fractions"][0],
        test_fraction=_SPLIT_PARAMS["fractions"][1],
        seed=_SPLIT_PARAMS["seed"], geology_indices=geo,
    )


def metrics_from(y_true, y_pred, graph_level: bool):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    sig = y_true > 0.01 * y_true.mean()
    ape = np.abs(y_pred[sig] - y_true[sig]) / np.abs(y_true[sig]) * 100
    ss_res = float(((y_pred - y_true) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    out = {
        "wmape": float(np.abs(y_pred - y_true).sum() / np.abs(y_true).sum() * 100),
        "median_ape": float(np.median(ape)),
        "r2": 1.0 - ss_res / ss_tot,
        "n": int(y_true.size),
    }
    if graph_level:
        out["spearman"] = float(spearmanr(y_true, y_pred).statistic)
        k = max(1, y_true.size // 10)
        out["top10_recall"] = len(
            set(np.argsort(y_true)[-k:]) & set(np.argsort(y_pred)[-k:])
        ) / k
    else:
        out["spearman"] = out["top10_recall"] = float("nan")
    return out


def find_ckpt(run_dir: Path):
    ckpts = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    assert len(ckpts) == 1, (
        f"expected exactly one best-*.ckpt in {run_dir}, found {len(ckpts)} "
        "(retrain residue? delete stale checkpoints)"
    )
    with open(run_dir / "checkpoints" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return ckpts[0], scaler


def mlp_feats(wells_list, idx, geo, feat_scaler=None):
    """Canonical-order flattened [x,y,depth,is_inj] + geology one-hot."""
    rows_wells, rows_flat = [], []
    for i in idx:
        w = wells_list[i]
        pos = np.stack([w["x"], w["y"], w["depth"]], axis=1).astype(np.float32)
        inj = (w["inj_rate"] > 0).astype(np.float32)
        order = canonical_well_order(torch.tensor(pos), torch.tensor(inj))
        feats = np.concatenate([pos[order], inj[order, None]], axis=1)
        rows_wells.append(feats)
    if feat_scaler is None:
        from sklearn.preprocessing import StandardScaler
        feat_scaler = StandardScaler().fit(np.concatenate(rows_wells))
    for i, feats in zip(idx, rows_wells):
        onehot = np.zeros(N_GEOLOGIES, dtype=np.float32)
        onehot[geo[i]] = 1.0
        rows_flat.append(np.concatenate(
            [feat_scaler.transform(feats).reshape(-1), onehot]).astype(np.float32))
    return np.stack(rows_flat), feat_scaler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-root", type=Path,
                   default=HERE / "runs" / "stage1")
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    stage = args.stage_root if args.stage_root.is_absolute() else (REPO / args.stage_root)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    seed_, fr_ = derive_split_params(stage)
    _SPLIT_PARAMS.update(seed=seed_, fractions=fr_)
    print(f"[cross-eval] split params: seed={seed_} fractions={fr_}")

    def run_dir(ds, tgt, var):
        logger = "baseline" if var in BASELINES else "geothermal_hetero_gnn"
        return stage / ds / tgt / var / f"seed{args.seed}" / logger / "run_00"

    # Incremental output with resume (audit F1): append per row, skip existing.
    out = stage / "cross_eval.csv"
    fieldnames = ["eval_dataset", "train_dataset", "target", "variant",
                  "wmape", "median_ape", "r2", "n", "spearman", "top10_recall"]
    done_keys: set[tuple] = set()
    existed = out.exists()
    if existed:
        with open(out) as f:
            done_keys = {(r["eval_dataset"], r["train_dataset"], r["target"],
                          r["variant"]) for r in csv.DictReader(f)}
        print(f"[cross-eval] resuming: {len(done_keys)} rows already present")
    out_f = open(out, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not existed:
        writer.writeheader()
        out_f.flush()

    def emit(row: dict):
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        out_f.flush()

    def is_done(eval_ds, train_ds, tgt, var):
        return (eval_ds, train_ds, tgt, var) in done_keys

    n_rows = len(done_keys)
    for eval_ds, h5_rel in DATASETS.items():
        h5 = REPO / h5_rel
        for tgt in TARGETS:
            graph_level = tgt.startswith("graph_")
            print(f"\n=== eval on {eval_ds} / {tgt} ===", flush=True)
            ids, tvals, wells_list, geo, wept = light_case_table(h5, tgt)
            tr_idx, va_idx, te_idx = split_for(tvals, geo)
            n_wells_set = {w.shape[0] for w in wells_list}
            assert len(n_wells_set) == 1, f"variable well counts: {n_wells_set}"
            n_wells = n_wells_set.pop()

            # Diagonal invariant (audit F3): the recomputed test split must
            # equal the artifact's test_predictions.csv case-id set.
            diag_csv = (run_dir(eval_ds, tgt, "gnn_default") / "plots"
                        / "test_predictions.csv")
            if diag_csv.exists():
                with open(diag_csv) as f:
                    artifact_ids = {r["case_id"] for r in csv.DictReader(f)}
                recomputed = {ids[i] for i in te_idx}
                assert recomputed == artifact_ids, (
                    f"{eval_ds}/{tgt}: recomputed test split differs from "
                    f"stage-1 artifact by {len(recomputed ^ artifact_ids)} ids"
                )

            # graphs per data-kwargs group, loaded lazily
            graph_cache: dict[tuple, tuple] = {}

            def graphs_for(kwargs_key):
                if kwargs_key not in graph_cache:
                    ne, nfm, enrich = kwargs_key
                    g, t = load_hetero_graphs(
                        h5, target=tgt, node_encoder=ne,
                        enrich_global_attr=enrich, node_features_mode=nfm,
                    )
                    graph_cache[kwargs_key] = (g, t)
                return graph_cache[kwargs_key]

            for train_ds in DATASETS:
                # ---- GNN variants ----
                for var in GNN_VARIANTS:
                    if is_done(eval_ds, train_ds, tgt, var):
                        continue
                    rd = run_dir(train_ds, tgt, var)
                    ckpt, scaler = find_ckpt(rd)
                    kw = peek_data_kwargs_from_checkpoint(ckpt)
                    graphs, _ = graphs_for((kw["node_encoder"],
                                            kw.get("node_features_mode", "full"),
                                            kw["enrich_global_attr"]))
                    test_graphs = [scaler.transform_graph(graphs[i]) for i in te_idx]
                    model = HeteroGNNRegressor.load_from_checkpoint(
                        str(ckpt), map_location=device)
                    model.eval()
                    y_true, y_pred, _ = evaluate_split(
                        model, test_graphs, scaler, args.batch_size, device)
                    m = metrics_from(y_true, y_pred, graph_level)
                    emit({"eval_dataset": eval_ds, "train_dataset": train_ds,
                          "target": tgt, "variant": var, **m})
                    n_rows += 1
                    print(f"  {var:<20} {train_ds:<15} wMAPE={m['wmape']:6.2f} "
                          f"R2={m['r2']:6.3f}", flush=True)
                    del model, test_graphs
                    torch.cuda.empty_cache()

                # ---- global_cnn (well-count agnostic) ----
                if not is_done(eval_ds, train_ds, tgt, "global_cnn"):
                    rd = run_dir(train_ds, tgt, "global_cnn")
                    ckpt, scaler = find_ckpt(rd)
                    graphs, _ = graphs_for(("cnn", "full", True))
                    cases = [build_case_arrays(graphs[i], int(geo[i]),
                                               scaler.target_scaler)
                             for i in te_idx]
                    ds_t = VolumeDataset([graphs[i] for i in te_idx], cases)
                    model = GlobalCNN3D.load_from_checkpoint(
                        str(ckpt), map_location=device)
                    model.eval()
                    preds, trues = [], []
                    with torch.no_grad():
                        for s in range(0, len(ds_t), args.batch_size):
                            items = [ds_t[i] for i in
                                     range(s, min(s + args.batch_size, len(ds_t)))]
                            vol = torch.stack([it[0] for it in items]).to(device)
                            pos = torch.stack([it[1] for it in items]).to(device)
                            inj = torch.stack([it[2] for it in items]).to(device)
                            y = torch.stack([it[3] for it in items])
                            mask = torch.stack([it[4] for it in items])
                            pred = model(vol, pos, inj).cpu()
                            if graph_level:
                                preds.append(pred.reshape(-1, 1))
                                trues.append(y.reshape(-1, 1))
                            else:
                                for r in range(pred.shape[0]):
                                    pm = mask[r].numpy()
                                    preds.append(pred[r].numpy()[pm, None])
                                    trues.append(y[r].numpy()[pm, None])
                    y_pred = scaler.inverse_targets(np.concatenate(preds))
                    y_true = scaler.inverse_targets(np.concatenate(trues))
                    m = metrics_from(y_true, y_pred, graph_level)
                    emit({"eval_dataset": eval_ds, "train_dataset": train_ds,
                          "target": tgt, "variant": "global_cnn", **m})
                    n_rows += 1
                    print(f"  {'global_cnn':<20} {train_ds:<15} "
                          f"wMAPE={m['wmape']:6.2f} R2={m['r2']:6.3f}", flush=True)
                    del model, ds_t, cases
                    torch.cuda.empty_cache()

                # ---- mlp (same well count only) ----
                tr_h5 = REPO / DATASETS[train_ds]
                with h5py.File(tr_h5, "r") as f:
                    first = next(iter(sorted(f.keys())))
                    tr_wells = f[first]["wells"].shape[0]
                if tr_wells != n_wells or is_done(eval_ds, train_ds, tgt, "mlp"):
                    continue
                rd = run_dir(train_ds, tgt, "mlp")
                ckpt, scaler = find_ckpt(rd)
                # refit the train dataset's feature scaler deterministically
                t_ids, t_tvals, t_wells, t_geo, _ = light_case_table(tr_h5, tgt)
                t_tr, _, _ = split_for(t_tvals, t_geo)
                _, feat_scaler = mlp_feats(t_wells, t_tr, t_geo)
                X, _ = mlp_feats(wells_list, te_idx, geo, feat_scaler)
                model = WellMLPBaseline.load_from_checkpoint(str(ckpt), map_location=device)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(X, device=device)).cpu().numpy()
                if graph_level:
                    y_pred_s, y_true_s = pred.reshape(-1, 1), tvals[te_idx].reshape(-1, 1)
                    y_pred = scaler.inverse_targets(y_pred_s)
                    y_true = y_true_s  # raw already
                else:
                    yp, yt = [], []
                    for r, i in enumerate(te_idx):
                        w = wells_list[i]
                        pos = np.stack([w["x"], w["y"], w["depth"]], axis=1).astype(np.float32)
                        inj = (w["inj_rate"] > 0).astype(np.float32)
                        order = canonical_well_order(torch.tensor(pos), torch.tensor(inj))
                        pm = inj[order] < 0.5
                        yp.append(pred[r][pm, None])
                        yt.append(wept[i][order][pm, None])
                    y_pred = scaler.inverse_targets(np.concatenate(yp))
                    y_true = np.concatenate(yt)
                m = metrics_from(y_true, y_pred, graph_level)
                emit({"eval_dataset": eval_ds, "train_dataset": train_ds,
                      "target": tgt, "variant": "mlp", **m})
                n_rows += 1
                print(f"  {'mlp':<20} {train_ds:<15} wMAPE={m['wmape']:6.2f} "
                      f"R2={m['r2']:6.3f}", flush=True)
                del model
                torch.cuda.empty_cache()

            del graph_cache
            gc.collect()

    out_f.close()
    print(f"\n{n_rows} total rows in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
