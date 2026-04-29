#!/usr/bin/bash
#SBATCH --job-name=GNN_ENSEMBLE
#SBATCH --output=/scratch/users/rwu4/geothermal_ensemble/logs/array_%A_%a.out
#SBATCH --error=/scratch/users/rwu4/geothermal_ensemble/logs/array_%A_%a.err
#SBATCH --array=0-9%5
#SBATCH --mem=48GB
#SBATCH -c 4
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -C GPU_SKU:L40S

set -euo pipefail

ml python/3.12.1 uv hdf5/1.14.4 openblas/0.3.20
export UV_PYTHON=$(which python3)
source ~/geothermal-pomdp/bin/activate

OUTPUT_ROOT="/scratch/users/rwu4/geothermal_ensemble/discountedrevenue"
mkdir -p "$OUTPUT_ROOT/logs"

python /home/users/rwu4/Geothermal_Graph_Surrogate/train.py \
  --h5-path /scratch/users/rwu4/geothermal_ensemble/discountedrevenue/compiled_CNN_revenue_newdiscount.h5 \
  --target graph_discounted_net_revenue \
  --seed 42 \
  --split-seed 42 \
  --run-id "$SLURM_ARRAY_TASK_ID" \
  --output-root "$OUTPUT_ROOT" \
  --cache-to-gpu \
  --gpu 0
