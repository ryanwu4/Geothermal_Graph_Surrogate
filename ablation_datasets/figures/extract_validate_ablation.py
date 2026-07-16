"""Validate the three ablation datasets and extract plotting arrays to NPZ.

Per case: well x, y, depth, inj_rate; final-year per-producer WEPT;
field energy total (t=-1); discounted revenue; geology index.
Runs the structural validation battery on the two new datasets as it goes.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ABL = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate/ablation_datasets")
OUT = ABL / "figures" / "dataset_summary_arrays.npz"

DATASETS = {
    "simple": dict(h5=ABL / "simple_2pair/seed_compiled.h5", n_wells=4, n_expect=2048),
    "hard": dict(h5=ABL / "hard_fulldepth/seed_compiled.h5", n_wells=12, n_expect=2048),
    "highperf": dict(h5=ABL / "highperf_cma/compiled.h5", n_wells=12, n_expect=3287),
}

S42_NORM = json.load(open(ABL / "highperf_cma/norm_config.json"))

arrays = {}
for name, spec in DATASETS.items():
    import re
    gm_path = spec["h5"].parent / ("case_geology_map.json")
    gm = json.load(open(gm_path))
    geo_of = {k: (v["geology_index"] if isinstance(v, dict) else v) for k, v in gm.items()}
    # scenario token sits immediately before _runNNNN_iterNNNN$ in both seed
    # ids and gated 'cmpl' panel ids; scenarios 71-85 <-> geology 0-14.
    SCEN_RE = re.compile(r"_(\d+)_run\d+_iter\d+$")

    def resolve_geo(cid):
        if cid in geo_of:
            return geo_of[cid]
        m = SCEN_RE.search(cid)
        return int(m.group(1)) - 71 if m else -1

    wx, wy, wd, winj = [], [], [], []
    wept_final, energy, revenue, geo_case = [], [], [], []
    n_bad = 0
    with h5py.File(spec["h5"], "r") as f:
        cases = sorted(f.keys())
        assert len(cases) == spec["n_expect"], (name, len(cases))
        for cid in cases:
            g = f[cid]
            w = g["wells"][:]
            # Wells can merge when two sampled wells project to the same grid
            # column (rare); compile then discovers fewer IsWell columns.
            if w.shape[0] != spec["n_wells"]:
                print(f"  [{name}] NOTE {cid}: {w.shape[0]} wells (merged column)")
            inj = w["inj_rate"] > 0
            ww = g["well_wept"][:]
            assert ww.shape == (w.shape[0], 30), (name, cid)
            final = ww[:, -1]
            # finite and non-negative required; ==0 is a legitimate label
            # (dead producer perforated above the productive zone).
            assert np.isfinite(final).all() and (final[~inj] >= 0).all(), (name, cid)
            n_bad += int((final[~inj] == 0).sum())
            wx.append(w["x"]); wy.append(w["y"]); wd.append(w["depth"])
            winj.append(inj)
            wept_final.append(final[~inj])
            energy.append(g["field_energy_production_total"][-1])
            revenue.append(float(g["field_discounted_net_revenue"][()]))
            geo_case.append(resolve_geo(cid))
    geo_case = np.array(geo_case)
    assert (geo_case >= 0).all(), (name, "unmapped geology cases")
    arrays[f"{name}_x"] = np.concatenate(wx).astype(np.float32)
    arrays[f"{name}_y"] = np.concatenate(wy).astype(np.float32)
    arrays[f"{name}_depth"] = np.concatenate(wd).astype(np.float32)
    arrays[f"{name}_isinj"] = np.concatenate(winj)
    arrays[f"{name}_wept_final"] = np.concatenate(wept_final).astype(np.float64)
    arrays[f"{name}_energy"] = np.array(energy, dtype=np.float64)
    arrays[f"{name}_revenue"] = np.array(revenue, dtype=np.float64)
    arrays[f"{name}_geo"] = geo_case
    counts = np.bincount(geo_case, minlength=15)
    print(f"[{name}] {len(cases)} cases OK | wells/case={spec['n_wells']} | "
          f"per-geo min/max={counts.min()}/{counts.max()} | "
          f"depth {arrays[f'{name}_depth'].min():.0f}-{arrays[f'{name}_depth'].max():.0f} | "
          f"revenue {np.min(revenue)/1e6:.0f}-{np.max(revenue)/1e6:.0f} M | "
          f"energy med {np.median(energy):.2e} | "
          f"dead producers {n_bad}/{sum(x.size for x in wept_final)} "
          f"({100*n_bad/sum(x.size for x in wept_final):.1f}%)")

# norm equality across all three
for sub in ("simple_2pair/seed_norm_config.json", "hard_fulldepth/seed_norm_config.json"):
    n = json.load(open(ABL / sub))
    same = json.dumps(n, sort_keys=True) == json.dumps(S42_NORM, sort_keys=True)
    print(f"[norm] {sub} identical to highperf norm: {same}")
    assert same

OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, **arrays)
print(f"saved {OUT}")
