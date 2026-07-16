#!/usr/bin/env python3
"""Regenerate a COMPLETE case_geology_map.json for an ablation dataset.

The gated_seed42 sidecar only covers its 256 seed cases, and the gated run's
'cmpl' panel case-ids don't match data.py's AL_RE — so train.py's JSON-map
path (which requires every case id present) falls back to slow runtime
resolution. The scenario token sits immediately before _runNNNN_iterNNNN$ in
every case-id family used here; scenarios 71-85 <-> geology_index 0-14.

Only touches the ablation_datasets/ copy — never the AL workspace sidecars.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import h5py

SCEN_RE = re.compile(r"_(\d+)_run\d+_iter\d+$")
SCENARIO_BASE = 71
N_GEOLOGIES = 15


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--map-path", type=Path, default=None,
                        help="Output map path (default: <h5 dir>/case_geology_map.json)")
    args = parser.parse_args()

    map_path = args.map_path or args.h5_path.parent / "case_geology_map.json"

    with h5py.File(args.h5_path, "r") as f:
        case_ids = sorted(f.keys())

    mapping: dict[str, dict[str, int]] = {}
    for cid in case_ids:
        m = SCEN_RE.search(cid)
        if not m:
            raise SystemExit(f"case id does not carry a scenario token: {cid}")
        geo = int(m.group(1)) - SCENARIO_BASE
        if not 0 <= geo < N_GEOLOGIES:
            raise SystemExit(f"scenario {m.group(1)} out of range for {cid}")
        mapping[cid] = {"geology_index": geo}

    if map_path.exists():
        old = json.loads(map_path.read_text())
        # Sanity: never contradict existing entries.
        for cid, entry in old.items():
            old_geo = entry["geology_index"] if isinstance(entry, dict) else entry
            if cid in mapping and mapping[cid]["geology_index"] != int(old_geo):
                raise SystemExit(f"disagreement with existing map on {cid}")
        backup = map_path.with_suffix(f".json.bak_{len(old)}")
        shutil.copy2(map_path, backup)
        print(f"backed up {len(old)}-entry map to {backup.name}")

    map_path.write_text(json.dumps(mapping, indent=1, sort_keys=True))
    counts = Counter(v["geology_index"] for v in mapping.values())
    print(f"wrote {len(mapping)} entries to {map_path}")
    print(f"per-geology counts: {sorted(counts.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
