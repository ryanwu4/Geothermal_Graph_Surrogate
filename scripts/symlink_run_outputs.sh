#!/usr/bin/env bash
# Symlink one staging run's IX outputs into a dedicated directory so they can
# be batched through preprocess_h5.py without sweeping in unrelated runs.
#
# The IX worker writes every realization across every staging run into the
# shared `h5s_dir_out` (default /scratch/users/rwu4/intersect_data/h5s_out).
# Each file is named `v2.5_<prefix>_<scenario>_run<NNNN>_iter<MMMM>.h5` where
# <prefix> matches the `--output-prefix` passed to cli_surrogate_array_prepare.jl
# (typically the staging dir's basename, e.g. `cmaes_20260505_142335`).
#
# This script collects all files matching that prefix and symlinks them under
# `<staging_dir>/h5s_out_links/`, then prints a ready-to-run preprocess_h5.py
# command. Optionally cross-checks against the staging dir's
# manifests/array_tasks.json to report any tasks that didn't produce output.
#
# Usage:
#   scripts/symlink_run_outputs.sh <staging_dir> [<source_h5_dir>]
#
# Example (matching the run mentioned by the user):
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

PREFIX="$(basename "$STAGING_DIR")"
DEST_DIR="$STAGING_DIR/h5s_out_links"
TASKS_JSON="$STAGING_DIR/manifests/array_tasks.json"

mkdir -p "$DEST_DIR"
echo "prefix:      $PREFIX"
echo "source:      $SRC_DIR"
echo "destination: $DEST_DIR"
echo

# Symlink every file matching v2.5_<prefix>_*.h5. Use find rather than glob
# so a missing match doesn't trip set -u, and so we get sorted output.
n_linked=0
n_already=0
while IFS= read -r src_path; do
    fname="$(basename "$src_path")"
    dest_path="$DEST_DIR/$fname"
    if [[ -L "$dest_path" || -e "$dest_path" ]]; then
        n_already=$((n_already + 1))
    else
        ln -s "$src_path" "$dest_path"
        n_linked=$((n_linked + 1))
    fi
done < <(find "$SRC_DIR" -maxdepth 1 -type f -name "v2.5_${PREFIX}_*.h5" | sort)

echo "linked:        $n_linked new symlinks"
echo "already there: $n_already symlinks (re-run is idempotent)"
echo "total in dest: $(find "$DEST_DIR" -maxdepth 1 -type l -name "*.h5" | wc -l)"

# Optional sanity check: how many output_file_names did the staging promise?
if [[ -f "$TASKS_JSON" ]]; then
    expected=$(python3 -c "
import json
with open('$TASKS_JSON') as f:
    payload = json.load(f)
print(len(payload.get('tasks', [])))
")
    actual=$(find "$DEST_DIR" -maxdepth 1 -type l -name "*.h5" | wc -l)
    echo "expected:      $expected tasks (per array_tasks.json)"
    echo
    if [[ "$actual" -lt "$expected" ]]; then
        # Identify the missing ones so the user can investigate.
        missing=$(python3 - <<EOF
import json, os
with open("$TASKS_JSON") as f:
    payload = json.load(f)
expected_names = {t["output_file_name"] for t in payload.get("tasks", [])}
present = set(os.listdir("$DEST_DIR"))
missing = sorted(expected_names - present)
print("\n".join(missing[:20]))
if len(missing) > 20:
    print(f"... and {len(missing) - 20} more")
EOF
)
        if [[ -n "$missing" ]]; then
            echo "WARN: $((expected - actual)) task output(s) not found in $SRC_DIR. First 20:" >&2
            echo "$missing" >&2
        fi
    fi
fi

echo
echo "Compile to a single HDF5 for analysis with:"
echo "  cd ~/Geothermal_Graph_Surrogate"
echo "  python preprocess_h5.py \\"
echo "      --input-dir $DEST_DIR \\"
echo "      --output-h5 $STAGING_DIR/compiled_${PREFIX}.h5 \\"
echo "      --norm-config trained/norm_config.json"
