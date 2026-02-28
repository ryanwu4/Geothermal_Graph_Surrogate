#!/usr/bin/env bash
# ===========================================================================
# render.sh — Parallel Render all Geothermal Visualization scenes.
#
# Usage:
#   ./render.sh [HDF5_FILE] [QUALITY]
#
# Arguments:
#   HDF5_FILE  Path to a raw v2.5_*.h5 file.
#              Default: ../data_test/v2.5_0111.h5
#   QUALITY    l (480p preview), m (720p), h (1080p production)
#              Default: l
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PYTHON="python"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

H5="${1:-${SCRIPT_DIR}/../data_test/v2.5_0111.h5}"
Q="${2:-h}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Geothermal Manim Parallel Renderer"
echo "  HDF5: ${H5}"
echo "  Quality: -q${Q}"
echo "  Logs: ${LOG_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "${SCRIPT_DIR}"

SCENES=(
    "scene_faults.py"
    "scene_permeability.py"
    "scene_astar.py"
    "scene_graph.py"
    "scene_master.py"
)

PIDS=()
LOGS=()

# --- Cleanup Handler ---
cleanup() {
    echo ""
    echo "🚨 Cancellation received. Terminating background renders..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "✔ Cleaned up. Exiting."
    exit 1
}

trap cleanup SIGINT SIGTERM

for scene_file in "${SCENES[@]}"; do
    if [ ! -f "$scene_file" ]; then continue; fi
    
    scene_class=$(grep '^class ' "${scene_file}" | head -1 | sed 's/class \([A-Za-z_]*\).*/\1/')
    log_file="${LOG_DIR}/${scene_class}.log"
    echo "▸ Starting ${scene_class} (${scene_file}) ..."
    
    # Run in background with logging
    "${CONDA_PYTHON}" -m manim render -q"${Q}" --disable_caching "${scene_file}" "${scene_class}" > "$log_file" 2>&1 &
    PIDS+=($!)
    LOGS+=("$log_file")
done

echo ""
echo "⏳ Waiting for all renders to complete..."

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    wait "$pid" || {
        # Check if it was killed by us (cleanup)
        if kill -0 "$pid" 2>/dev/null; then
            echo "❌ Render FAILED for scene ${SCENES[$i]}. Check: ${LOGS[$i]}"
            FAILED=1
        fi
    }
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "✔ All scenes rendered successfully in parallel."
    echo "Videos available in media/videos/"
else
    echo ""
    echo "⚠ Some renders failed. See errors above."
    exit 1
fi
