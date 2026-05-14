"""Create a 1000-case benchmark subset of current_compiled.h5.

- 200 cases from geo 8.
- 800 cases distributed across the other 14 geologies (~57 each).
- The subset H5 uses h5py.ExternalLink so no physics_tensors are duplicated;
  the file is ~tens of KB.

The subset preserves the top-level file attrs that the surrogate's data loader
reads (energy_price, discount_rate, etc.).

Outputs:
  benchmark.h5            — small file of external links into current_compiled.h5
  benchmark_manifest.json — case_id -> geology_index for the sampled cases
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import h5py

WORKSPACE = Path("/home/rwu4/omv_geothermal/geothermal_active_learning/local_workspace")
SOURCE_H5 = WORKSPACE / "current_compiled.h5"
HERE = Path(__file__).resolve().parent
MAP_JSON = HERE / "case_geology_map.json"
OUT_H5 = HERE / "benchmark.h5"
OUT_MANIFEST = HERE / "benchmark_manifest.json"

GEO8_TARGET = 200
TOTAL = 1000
OTHER_GEOS = [g for g in range(15) if g != 8]
PER_OTHER = (TOTAL - GEO8_TARGET) // len(OTHER_GEOS)  # 57
EXTRA = (TOTAL - GEO8_TARGET) - PER_OTHER * len(OTHER_GEOS)  # 2 leftover

SEED = 42


def main() -> None:
    case_map = json.loads(MAP_JSON.read_text())

    # Group case ids by geology
    by_geo: dict[int, list[str]] = defaultdict(list)
    for cid, info in case_map.items():
        by_geo[info["geology_index"]].append(cid)

    rng = random.Random(SEED)
    chosen: list[tuple[str, int]] = []

    # geo 8
    g8 = sorted(by_geo[8])
    rng.shuffle(g8)
    take = min(GEO8_TARGET, len(g8))
    chosen.extend((cid, 8) for cid in g8[:take])
    print(f"geo  8: took {take}/{len(g8)}")

    # other geos
    for i, g in enumerate(OTHER_GEOS):
        cids = sorted(by_geo[g])
        rng.shuffle(cids)
        n_take = PER_OTHER + (1 if i < EXTRA else 0)
        n_take = min(n_take, len(cids))
        chosen.extend((cid, g) for cid in cids[:n_take])
        print(f"geo {g:2d}: took {n_take}/{len(cids)}")

    print(f"Total chosen: {len(chosen)}")
    counts = Counter(g for _, g in chosen)
    print(f"Distribution: {dict(sorted(counts.items()))}")

    # Build benchmark.h5 with external links
    if OUT_H5.exists():
        OUT_H5.unlink()
    with h5py.File(SOURCE_H5, "r") as src, h5py.File(OUT_H5, "w") as dst:
        # Copy top-level attrs (the loader reads these for discount rate, energy price, etc.)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        # Create external links per chosen case
        for cid, g in chosen:
            dst[cid] = h5py.ExternalLink(str(SOURCE_H5.resolve()), f"/{cid}")

    print(f"Wrote {OUT_H5} (size={OUT_H5.stat().st_size} bytes)")

    manifest = {cid: g for cid, g in chosen}
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {OUT_MANIFEST}")

    # Sanity: open via external links and verify groups resolve
    with h5py.File(OUT_H5, "r") as f:
        keys = list(f.keys())
        assert len(keys) == len(chosen)
        sample = f[keys[0]]
        assert "physics_tensors" in sample
        assert sample["wells"].shape == (12,)
    print("Sanity check passed — external links resolve.")


if __name__ == "__main__":
    main()
