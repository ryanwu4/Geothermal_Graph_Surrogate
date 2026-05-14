"""Build a case_id -> geology_index map for current_compiled.h5.

Two complementary sources, no fingerprinting needed:
1. Bootstrap cases (compiled keys like `v2.5_NNNN`): the file number N maps via
   filenum_to_scenario_mapping.csv to a Scenario, and the Scenario maps via
   geologies_full_local.json to a geology_index. (v2.5 reuses v2.4's file
   numbering — only the .h5 contents differ.)
2. AL-acquired cases (compiled keys like
   `v2.5_al_...iterNNNN_<scenario>_run<geo*10000+rid>_iterMM`): geology_index
   is encoded directly in the run number as `run_num // 10000`. We also
   cross-check the embedded scenario id against the geologies config.

We additionally cross-validate against a cheap rounded-log10-mean fingerprint
of PermZ to confirm the two methods agree.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

WORKSPACE = Path("/home/rwu4/omv_geothermal/geothermal_active_learning/local_workspace")
COMPILED_H5 = WORKSPACE / "current_compiled.h5"
GEO_CFG = Path("/home/rwu4/omv_geothermal/geothermal_active_learning/configs/geologies_full_local.json")
FILENUM_CSV = Path("/home/rwu4/omv_geothermal/GeologicalSimulationWrapper.jl/filenum_to_scenario_mapping.csv")
OUT_JSON = Path(__file__).resolve().parent / "case_geology_map.json"

BOOT_RE = re.compile(r"^v2\.5_(\d{4})$")
AL_RE = re.compile(r".*_iter\d+_(\d+)_run(\d+)_iter\d+$")


def main() -> None:
    # scenario -> geology_index
    geo_cfg = json.loads(GEO_CFG.read_text())
    scenario_to_geo: dict[int, int] = {}
    for entry in geo_cfg["geologies"]:
        scenario_to_geo[int(entry["scenario"])] = int(entry["geology_index"])
    print(f"Geology config: {len(scenario_to_geo)} (scenario -> geo_idx) mappings")

    # filenum -> scenario (from v2.4 mapping; reused by v2.5)
    filenum_to_scenario: dict[int, int] = {}
    with open(FILENUM_CSV) as f:
        for row in csv.DictReader(f):
            filenum_to_scenario[int(row["Num"])] = int(row["Scenario"])
    print(f"Filenum CSV: {len(filenum_to_scenario)} (filenum -> scenario) mappings")

    case_to_geo: dict[str, dict] = {}
    n_bootstrap = n_al = n_unmapped = 0

    with h5py.File(COMPILED_H5, "r") as f:
        case_ids = sorted(f.keys())
        for cid in case_ids:
            geo_idx = None
            source = None

            m_boot = BOOT_RE.match(cid)
            m_al = AL_RE.match(cid)
            if m_boot is not None:
                filenum = int(m_boot.group(1))
                scen = filenum_to_scenario.get(filenum)
                if scen is not None:
                    geo_idx = scenario_to_geo.get(scen)
                    source = "filenum_csv"
            elif m_al is not None:
                scen_from_name = int(m_al.group(1))
                run_num = int(m_al.group(2))
                geo_idx = run_num // 10000
                # Cross-check: scenario embedded in name should match the geo's scenario.
                expected_scen = next(
                    (s for s, g in scenario_to_geo.items() if g == geo_idx), None
                )
                if expected_scen is not None and expected_scen != scen_from_name:
                    print(
                        f"  WARNING: case {cid}: scenario {scen_from_name} in name but geo {geo_idx} -> scenario {expected_scen}"
                    )
                source = "al_runid"

            if geo_idx is None:
                n_unmapped += 1
                continue

            case_to_geo[cid] = {"geology_index": int(geo_idx), "source": source}
            if source == "filenum_csv":
                n_bootstrap += 1
            else:
                n_al += 1

    print(f"Bootstrap cases mapped: {n_bootstrap}")
    print(f"AL cases mapped:        {n_al}")
    print(f"Unmapped cases:         {n_unmapped}")
    counts = Counter(v["geology_index"] for v in case_to_geo.values())
    print(f"Cases per geology: {sorted(counts.items())}")

    OUT_JSON.write_text(json.dumps(case_to_geo, indent=2))
    print(f"Wrote {OUT_JSON} ({len(case_to_geo)} cases)")

    # Cross-validate via rounded-log10-mean PermZ fingerprint.
    print("\nCross-validating against PermZ fingerprint...")
    with h5py.File(COMPILED_H5, "r") as f:
        # For each geo, compute median log10(PermZ) over up to 5 sample cases.
        per_geo_fp: dict[int, list[float]] = {}
        per_geo_sample: dict[int, list[str]] = {}
        for cid, info in case_to_geo.items():
            g = info["geology_index"]
            if len(per_geo_sample.setdefault(g, [])) >= 5:
                continue
            per_geo_sample[g].append(cid)
        for g, cids in per_geo_sample.items():
            for cid in cids:
                permz = f[cid]["physics_tensors"]["PermZ"][:]
                valid = f[cid]["physics_tensors"]["valid_mask"][:] > 0.5
                vals = permz[valid]
                fp = float(np.mean(vals)) if vals.size > 0 else float("nan")
                per_geo_fp.setdefault(g, []).append(fp)

    # Report per-geo fp range; should be tight within a geology
    bad = []
    for g in sorted(per_geo_fp):
        vals = np.array(per_geo_fp[g])
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        if spread > 0.05:
            bad.append((g, spread))
        print(f"  geo {g:2d}: fp mean={np.nanmean(vals):.3f}, spread={spread:.4f}")
    if bad:
        print(f"  WARNING: geologies with large fp spread (>0.05): {bad}")
    else:
        print("  OK: all geologies have tight fingerprint clusters.")


if __name__ == "__main__":
    main()
