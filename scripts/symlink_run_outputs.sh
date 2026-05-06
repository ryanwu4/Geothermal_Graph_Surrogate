#!/usr/bin/env bash
# Symlink one staging run's IX outputs into a dedicated directory so they can
# be batched through preprocess_h5.py without sweeping in unrelated runs.
#
# The IX worker writes every realization across every staging run into the
# shared `h5s_dir_out` (default /scratch/users/rwu4/intersect_data/h5s_out).
# This script reads the staging dir's `manifests/array_tasks.json` —
# authoritative for which output_file_names belong to this run — and symlinks
# each into a per-run `h5s_out_links/` subdirectory.
#
# Usage:
#   scripts/symlink_run_outputs.sh <staging_dir> [<source_h5_dir>]
#
# Example:
#   scripts/symlink_run_outputs.sh \
#       /scratch/users/rwu4/intersect_data/gradient_calibrate_patched/cmaes_20260505_142335

set -euo pipefail

STAGING_DIR="${1:-}"
SRC_DIR="${2:-/scratch/users/rwu4/intersect_data/h5s_out}"

if [[ -z "$STAGING_DIR" ]]; then
    echo "Usage: $0 <staging_dir> [<source_h5_dir>]" >&2
    exit 1
fi
if [[ ! -d "$STAGING_DIR" ]]; then
    echo "ERROR: staging dir not found: $STAGING_DIR" >&2
    exit 1
fi
if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: source dir not found: $SRC_DIR" >&2
    exit 1
fi

TASKS_JSON="$STAGING_DIR/manifests/array_tasks.json"
if [[ ! -f "$TASKS_JSON" ]]; then
    echo "ERROR: tasks JSON not found at $TASKS_JSON" >&2
    echo "       (expected at <staging_dir>/manifests/array_tasks.json — make sure" >&2
    echo "        the path you passed is the staging dir, not a parent.)" >&2
    exit 1
fi

DEST_DIR="$STAGING_DIR/h5s_out_links"
mkdir -p "$DEST_DIR"

echo "staging_dir: $STAGING_DIR"
echo "source:      $SRC_DIR"
echo "destination: $DEST_DIR"
echo

ml python/3.12.1 uv hdf5/1.14.4 openblas/0.3.20
export UV_PYTHON=$(which python3)
source ~/geothermal-pomdp/bin/activate

# Use python to walk the tasks list — bash + jq isn't always available, but
# python3 is on every Sherlock node we use.
python3 - "$TASKS_JSON" "$SRC_DIR" "$DEST_DIR" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

tasks_json, src_dir, dest_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

with open(tasks_json) as f:
    payload = json.load(f)
tasks = payload.get("tasks", [])

n_total = len(tasks)
n_linked = 0
n_already = 0
missing = []

for t in tasks:
    name = t.get("output_file_name", "")
    if not name:
        continue
    src = src_dir / name
    dst = dest_dir / name
    if not src.exists():
        missing.append(name)
        continue
    if dst.exists() or dst.is_symlink():
        n_already += 1
    else:
        os.symlink(src.resolve(), dst)
        n_linked += 1

print(f"linked:        {n_linked} new symlinks")
print(f"already there: {n_already} symlinks (re-run is idempotent)")
print(f"total in dest: {n_linked + n_already} of {n_total} tasks")
if missing:
    print()
    print(f"WARN: {len(missing)} task output(s) not found in {src_dir}.")
    print("First 20 missing output_file_names:")
    for name in missing[:20]:
        print(f"  {name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
PYEOF

echo
echo "Compile to a single HDF5 for analysis with:"
echo "  cd ~/Geothermal_Graph_Surrogate"
echo "  python preprocess_h5.py \\"
echo "      --input-dir $DEST_DIR \\"
echo "      --output-h5 $STAGING_DIR/compiled.h5 \\"
echo "      --norm-config trained/norm_config.json"
