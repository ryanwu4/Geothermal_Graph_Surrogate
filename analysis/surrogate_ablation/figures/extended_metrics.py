"""Extended test metrics per stage-1 run from test_predictions.csv:
median APE, wMAPE, Spearman rank corr + top-decile recall (graph targets).
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate/analysis/surrogate_ablation/runs/stage1")
DATASETS = ["simple_2pair", "hard_fulldepth", "highperf_cma"]
TARGETS = ["graph_energy_total", "graph_discounted_net_revenue", "node_wept_final"]
VARIANTS = ["gnn_default", "gnn_dist_edge", "gnn_type_only_nodes",
            "gnn_svd_edge", "mlp", "global_cnn"]

rows = []
for ds in DATASETS:
    for tgt in TARGETS:
        for var in VARIANTS:
            base = ROOT / ds / tgt / var / "seed42"
            logger = "baseline" if var in ("mlp", "global_cnn") else "geothermal_hetero_gnn"
            plots = base / logger / "run_00" / "plots"
            pred_csv = plots / "test_predictions.csv"
            if not pred_csv.exists():
                continue
            with open(pred_csv) as f:
                recs = list(csv.DictReader(f))
            y_true = np.array([float(r["y_true"]) for r in recs])
            y_pred = np.array([float(r["y_pred"]) for r in recs])
            summary = json.loads((plots / "metrics_summary.json").read_text())
            m = summary["splits"]["test"]["metrics"]

            sig = y_true > 0.01 * y_true.mean()
            ape = np.abs(y_pred[sig] - y_true[sig]) / np.abs(y_true[sig]) * 100
            med_ape = float(np.median(ape))
            wmape = float(np.abs(y_pred - y_true).sum() / np.abs(y_true).sum() * 100)

            rho = top_recall = float("nan")
            if tgt.startswith("graph_"):
                rho = float(spearmanr(y_true, y_pred).statistic)
                k = max(1, len(y_true) // 10)
                top_t = set(np.argsort(y_true)[-k:])
                top_p = set(np.argsort(y_pred)[-k:])
                top_recall = len(top_t & top_p) / k

            rows.append({
                "dataset": ds, "target": tgt, "variant": var,
                "mape": round(m["mape"], 3), "r2": round(m["r2"], 4),
                "median_ape": round(med_ape, 3), "wmape": round(wmape, 3),
                "spearman": round(rho, 4) if rho == rho else "",
                "top10_recall": round(top_recall, 4) if top_recall == top_recall else "",
                "n_test": len(y_true),
            })

out = ROOT / "extended_metrics.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}\n")

hdr = f"{'variant':<20}" + "".join(f"{d[:8]:>26}" for d in DATASETS)
for tgt in TARGETS:
    print(f"== {tgt} ==  (medAPE% / wMAPE% / spearman)")
    print(hdr)
    for var in VARIANTS:
        line = f"{var:<20}"
        for ds in DATASETS:
            r = next((x for x in rows if x["dataset"] == ds and x["target"] == tgt
                      and x["variant"] == var), None)
            if r:
                sp = f"{r['spearman']:.3f}" if r["spearman"] != "" else "  -  "
                line += f"{r['median_ape']:>8.1f}/{r['wmape']:>5.1f}/{sp:>6}"
            else:
                line += f"{'-':>26}"
        print(line)
    print()
