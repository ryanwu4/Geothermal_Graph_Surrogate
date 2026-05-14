# Geothermal GNN — Network Architecture & Usage Guide

## Overview

This project uses a **heterogeneous graph neural network (Hetero-GNN)** to predict field-level energy production from geothermal well configurations. Wells are modeled as nodes in a graph, with edges representing subsurface geological connectivity via **k-nearest neighbors**. Edge and node features are dynamically extracted by **3D CNNs** that crop and process local geological volumes around each well and each well pair. The model is built with [PyTorch Geometric](https://pyg.org) and trained via [PyTorch Lightning](https://lightning.ai).

The entire pipeline — from continuous well coordinates through physics interpolation, CNN feature extraction, and GNN prediction — is **end-to-end differentiable**, enabling gradient-based optimization of well placement.

> **Note on defaults (May 2026).** After a deep investigation into a geo-8 OOD
> failure mode, the model defaults changed substantially. Most prominently:
> `--loss` is now MSE (was Huber), the edge CNN uses `GroupNorm` + a raw-means
> bypass (was BatchNorm), node features are produced by a 3D CNN around each
> well (`--node-encoder cnn`, was the 25-d hand-engineered profile), and
> `--enrich-global-attr` adds reservoir-mean physics stats to the graph-level
> input. See `analysis/geo8_benchmark/FINDINGS.md` for the diagnostic story
> and benchmark numbers.

---

## Network Architecture

### Graph Structure

Each simulation run is encoded as a single **heterogeneous graph**:

| Component | Description |
|---|---|
| **Nodes** | One node per well (injector or producer) |
| **Edge types** | 4 typed relations: `inj→inj`, `ext→ext`, `inj→ext`, `ext→inj` |
| **Edge source** | K-nearest neighbors (k=2 per type) based on 3D Euclidean distance |

### Node Features

The node encoder is selected by `--node-encoder` (default: `cnn`):

**`--node-encoder cnn` (default)** — 9 base scalars + a learned 32-d CNN embedding:

| Index | Feature | Source |
|---|---|---|
| 0 | Injection rate | Well operating parameter |
| 1 | Perf top Z-index | Well geometry (= max(0, depth − n_layers + 1)) |
| 2–4 | Permeability (x, y, z) at well cell | Reservoir property at well |
| 5 | Porosity at well cell | Reservoir property at well |
| 6 | Initial temperature at well cell | Reservoir state |
| 7 | Initial pressure at well cell | Reservoir state |
| 8 | n_layers (perforation length) | Well geometry |

Plus a `PhysicsNodeSlabCNN` 32-d embedding produced by cropping a small 3D slab
around each well (full-Z, ±3 in XY by default), concatenated with the 9 scalars.
The CNN uses GroupNorm (not BatchNorm — see FINDINGS.md) and a raw-channel-means
bypass that preserves absolute-magnitude signal across geologies.

**`--node-encoder profile` (legacy)** — 8 base scalars (with `depth` instead of
`perf_top` and *no* `n_layers`) + 25 hand-engineered `well_vertical_profile`
statistics (6 properties × {mean, min, max, std} + n_layers) = 33 dims total.

**`--node-encoder hybrid`** — 33-d legacy profile features *plus* the CNN
embedding. Mostly redundant with profile alone (see ablation study) but
available for experiments.

### Edge Features (dynamically computed by 3D CNN)

Edge features are **not hardcoded**. A `PhysicsSlabCNN` dynamically extracts them:

1. For each well pair (edge), a **3D sub-volume** of the geology grid is cropped using differentiable `grid_sample` interpolation
2. The crop is resized to a fixed 32×32×16 volume and concatenated with 3D Gaussian heatmaps marking each well's position
3. A **3D CNN** (Conv3d → norm → activation → MaxPool) processes the volume into a latent vector
4. The latent vector is concatenated with the Euclidean distance Δs between the wells, and (when `--edge-raw-means` is set, default) with a 16-d projection of per-channel slab means computed pre-CNN
5. An MLP compresses the result to the configured `latent_edge_dim` (default: 32)

**Normalization**: `--edge-norm groupnorm` (default) preserves per-instance
absolute scale across the batch. The earlier BatchNorm default stripped the
absolute-magnitude signal that distinguishes outlier geologies — switching to
GroupNorm + a raw-means bypass was the central fix for the geo-8 OOD problem.

**Physics channels** processed by the CNN:
- PermX, PermY, PermZ (log-transformed, min-max normalized)
- Porosity, Temperature0, Pressure0 (min-max normalized)
- valid_mask (binary rock/void indicator)

### Global Features

The graph-level features fed to the prediction head are controlled by
`--enrich-global-attr` (default: enabled):

**`--enrich-global-attr` (default)** — 8 dimensions:

| Index | Feature |
|---|---|
| 0 | Well count |
| 1–3 | Reservoir-mean PermX / PermY / PermZ (over valid voxels) |
| 4 | Reservoir-mean porosity |
| 5 | Reservoir-mean initial temperature |
| 6 | Reservoir-mean initial pressure |
| 7 | Anisotropy (mean PermZ / mean PermX) |

These give the prediction head a direct geology fingerprint that bypasses the
GNN message-passing — disproportionately helpful for OOD geologies whose
physics distribution differs from the cohort.

**`--no-enrich-global-attr`** — legacy 1-d `global_attr = [n_wells]`.

### Model Architecture (`HeteroGNNRegressor`)

```
Input: HeteroData graph
  │
  ├── PhysicsSlabExtractor: crop 3D sub-volumes per edge
  │     → PhysicsSlabCNN: Conv3d(8→32→64→128) + MLP → edge_attr
  │
  ├── [NNConv × 4 edge types] ──► HeteroConv (aggr="sum")
  │         │
  │    LayerNorm → GELU → Dropout
  │         │
  │    Residual connection (projected if dim changes)
  │         │
  └── Repeat × num_layers (default: 2)
  │
  ├── Graph-level readout (for graph_energy_total):
  │     concat(global_add_pool, global_mean_pool, global_max_pool)
  │     + global features (well count)
  │     → FNN: [3H+G → 2H → H → H/2 → H/4 → 1]
  │
  └── Node-level readout (for node_wept / node_tp_final):
        → FNN: [H → H → H/2 → H/4 → output_dim]
```

**Key design choices:**
- **NNConv** with edge-conditioned message passing: a 2-layer MLP transforms the CNN-generated edge features into a weight matrix for each edge
- **Heterogeneous convolutions** (`HeteroConv`) apply separate NNConv operators per edge type, then aggregate via summation
- **3D CNN edge features** replace legacy A\* pathfinding, enabling fully differentiable edge computation
- **Triple pooling** (sum + mean + max) captures both magnitude and distribution information across the well network
- **GELU** activation throughout (smoother than ReLU)
- **LayerNorm** per GNN layer for stable training

### Default Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| Hidden dim | 32 | |
| Latent edge dim | 32 | |
| Latent node dim | 32 | (when `--node-encoder cnn`) |
| Edge CNN channels | 8 → 16 → 32 → 32 | |
| Node CNN channels | 16 → 32 → 32 | Z-stride only; preserves 7×7 XY |
| Num GNN layers | 2 | |
| Dropout | 0.0 | |
| Learning rate | 3e-4 | |
| Weight decay | 1e-2 | |
| **Loss** | **MSE** | Was Huber(δ=1.0); switched in May 2026 |
| **Edge norm** | **GroupNorm** | Was BatchNorm3d |
| **Edge raw-means bypass** | **on** | New (see FINDINGS) |
| **Node encoder** | **cnn** | Was hand-coded `profile` |
| **`enrich_global_attr`** | **on** | New (was just `[n_wells]`) |
| **Slab Z extent (node)** | **full** | covers whole active reservoir, not just perforation |
| Optimizer | AdamW | |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) | |
| Early stopping | patience=30 on val_loss | Used by AL workflow |
| TF32 matmul precision | `"high"` | ~2× throughput on Ampere+ |

---

## Data Pipeline

### Step 1: Preprocess Raw HDF5 → Normalized Dataset

```bash
python preprocess_h5.py \
    --input-dir data/ \
    --output-h5 compiled_full_CNN.h5 \
    --norm-config norm_config.json \
    --economics-config configs/economics_discounted_revenue.json
```

This processes each raw simulation HDF5 file:
1. Extracts well locations, injection rates, and reservoir properties
2. Computes vertical profile statistics (25 features per well)
3. Identifies valid rock cells and computes Z-axis depth cutoff (first layer where ≥95% of cells are inactive)
4. Log-transforms permeability, then min-max normalizes all physics channels to [0, 1]
5. Saves the processed 3D physics tensors and well tables into a compiled HDF5

For discounted-revenue training, preprocessing also computes:
- `field_discounted_net_revenue`: discounted sum of net energy revenue from annual field rates
- uses `ENERGY_PRICE`, `NOMINAL_DISCOUNT_RATE`, and `INFLATION_RATE` loaded from `--economics-config`
    (real discount rate computed via the Fisher equation)

**First run**: computes global normalization statistics across all files and writes `norm_config.json`.
**Subsequent runs**: reuses `norm_config.json` for consistent normalization.

Use `--compute-only` to only compute normalization stats without building the H5.

### Step 2: Train

```bash
python train.py \
    --h5-path compiled_full_CNN.h5 \
    --target graph_discounted_net_revenue \
    --stratified-split \
    --gpu 0 \
    --cache-to-gpu
```

All Tier-A/B fixes (geology-aware stratified split, MSE loss, GroupNorm edge CNN,
raw-means bypasses, CNN node encoder with full-Z slab, enriched global_attr,
TF32 matmul precision) are on by default. To disable any one:

| Flag to disable | Effect |
|---|---|
| `--loss huber` | Switch back to Huber loss (note: prediction-floor artifact). |
| `--edge-norm batchnorm` | Revert edge CNN to BatchNorm3d. |
| `--no-edge-raw-means` | Disable edge raw-means bypass. |
| `--node-encoder profile` | Use the legacy 25-d hand-engineered profile. |
| `--node-z-extent perforation` | Resample slab Z to exactly the well perforation. |
| `--no-enrich-global-attr` | Use `[n_wells]` only. |
| `--no-stratified-split` | Use a plain random train/val/test split. |

### Geology-aware stratified split

`--stratified-split` is on by default. The per-case geology index is resolved
in this order:

1. `case_geology_map.json` adjacent to the H5, if present. This file is written
   automatically by `preprocess_h5.py` / `compile_minimal_geothermal_h5.py` at
   the end of compilation, so freshly-compiled datasets get it for free.
2. Otherwise resolved at training time from case_id patterns:
   - AL-acquired cases (`..._iter\d+_<scen>_run<runnum>_iter\d+`) →
     `geology_index = runnum // 10000`.
   - Bootstrap cases (`v2.5_NNNN` / `v2.4_NNNN`) → looked up via
     `filenum_to_scenario_mapping.csv` (searched in surrogate repo root,
     surrogate `configs/`, then the Julia repo) and a `geologies_full*.json`.
   - Otherwise a fingerprint match on a representative case's PermZ field.

If geology indices can't be resolved for every case, the split falls back to
target-only quantile stratification.

**With top-k% withholding** (removes top 10% of runs by energy production):
```bash
python train.py \
    --h5-path compiled_full_CNN.h5 \
    --target graph_energy_total \
    --withhold-top-pct 10
```

Training produces:
- Best model checkpoint: `lightning_logs/.../checkpoints/best-*.ckpt`
- Fitted scaler: `lightning_logs/.../checkpoints/scaler.pkl`
- Evaluation plots: scatter, error distributions, loss curves, extreme-error graphs
- (If withholding): `withheld_runs.json` + `withholding_histogram.png`

Additional flags:
- `--cache-to-gpu`: preload all training data onto GPU memory
- `--gpu N`: select a specific GPU device
- `--ablate GROUP [GROUP ...]`: zero out feature groups for ablation studies

**Deep ensemble training (sbatch array friendly)**

Use `--run-id` to create distinct ensemble members with different random seeds while
keeping a fixed train/val/test split via `--split-seed`.

```bash
python train.py \
    --h5-path compiled_CNN_revenue_newdiscount.h5 \
    --target graph_discounted_net_revenue \
    --seed 42 \
    --split-seed 42 \
    --run-id 0 \
    --output-root trained/ensemble/discountedrevenue \
    --gpu 0
```

For a 10-member sbatch array, set `--run-id $SLURM_ARRAY_TASK_ID` and keep
`--split-seed` constant to preserve the same split across members.

### Step 3: Differentiable Inference & Well Placement Optimization

Run gradient-based optimization of well placement on a custom JSON-defined well configuration:

```bash
python run_differentiable_inference.py \
    --config example_inference.json \
    --optimization-steps 100 \
    --learning-rate 0.5
```

The JSON configuration file specifies the geology, model, and initial well layout:

```json
{
    "geology_h5_file": "data_test/v2.5_0111.h5",
    "checkpoint": "trained/.../checkpoints/best-*.ckpt",
    "scaler_path": "trained/.../checkpoints/scaler.pkl",
    "norm_config": "trained/norm_config.json",
    "objective_target": "graph_discounted_net_revenue",
    "wells": [
        {"x": 15, "y": 10, "depth": 50, "type": "injector"},
        {"x": 19, "y": 14, "type": "producer"}
    ]
}
```

**JSON fields:**
| Field | Required | Description |
|---|---|---|
| `geology_h5_file` | Yes | Path to raw simulation HDF5 with geology data |
| `checkpoint` | Yes | Path to trained model `.ckpt` file |
| `scaler_path` | Yes | Path to `scaler.pkl` saved during training |
| `norm_config` | No | Path to `norm_config.json` (default: `norm_config.json`) |
| `objective_target` | No | Graph target to optimize: `graph_energy_total`, `graph_energy_rate`, or `graph_discounted_net_revenue` |
| `wells` | Yes | Array of well definitions (see below) |

**Well fields:**
| Field | Required | Description |
|---|---|---|
| `x` | Yes | Grid X coordinate |
| `y` | Yes | Grid Y coordinate |
| `type` | Yes | `"injector"` or `"producer"` |
| `depth` | No | Max perforation depth in Z layers (default: full reservoir depth) |

The script:
1. Reads raw geology from the HDF5 file and normalizes it on-the-fly using `norm_config.json`
2. Builds a `PhysicsContext` with the normalized 3D tensors
3. Constructs a graph from the well layout and runs it through the trained model
4. Optimizes well (X, Y, Z) coordinates via Adam gradient ascent to maximize the configured graph-level objective
5. **Feasible direction projection**: gradients pointing outside the grid domain are projected along the boundary; Adam momentum is flushed for boundary-violating components
6. Outputs a 2D trajectory plot (wells over log-PermX background) alongside an objective-vs-iteration curve to `plots/`

All plots are saved with timestamps to prevent overwriting.

---

## Prediction Targets

| Target flag | Level | Output dim | Description |
|---|---|---|---|
| `graph_energy_total` | Graph | 1 | Total field energy production (cumulative, final timestep) |
| `graph_energy_rate` | Graph | 1 | Field energy production rate (first timestep) |
| `graph_discounted_net_revenue` | Graph | 1 | Discounted net energy revenue, computed from annual `(production - injection)` energy rates |
| `node_wept` | Node | 1 | Well energy production timeseries per extractor well |
| `node_tp_final` | Node | 6 | Penultimate-timestep T/P profile statistics (mean/min/max for temp & pressure) |

---

## Feature Ablation

Zero out feature groups for ablation studies using `--ablate`:

```bash
python train.py --h5-path compiled_full_CNN.h5 --ablate base_perm base_thermo
```

| Group | Type | Columns zeroed |
|---|---|---|
| `base_perm` | Node | 2–4 (perm x/y/z) |
| `base_thermo` | Node | 6–7 (temp0, press0) |

> **Note:** Edge feature ablation is no longer applicable since edge features are dynamically generated by the 3D CNN from the full physics volume rather than precomputed.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate geothermal-pomdp
```

---

## Project Structure

```
├── environment.yml                     # Conda environment spec
├── train.py                            # Training entry point
├── preprocess_h5.py                    # Raw HDF5 → compiled normalized dataset
├── run_differentiable_inference.py     # Differentiable well placement optimizer
├── compile_minimal_geothermal_h5.py    # Well extraction utilities
├── example_inference.json              # Sample JSON config for inference
├── configs/
│   └── economics_discounted_revenue.json  # Economics constants for discounted-revenue target
├── norm_config.json                    # Global normalization parameters
│
├── geothermal/                         # Core package
│   ├── __init__.py
│   ├── model.py                        # HeteroGNNRegressor + constants
│   ├── physics_slab.py                 # PhysicsSlabCNN + 3D cropper
│   ├── data.py                         # Loading, scaling, splitting, withholding
│   └── evaluation.py                   # Metrics, plotting, error analysis
│
├── trained/                            # Training outputs (checkpoints, plots)
│   ├── norm_config.json
│   └── withheld_0p_totalenergy/
│       ├── checkpoints/
│       │   ├── best-*.ckpt
│       │   └── scaler.pkl
│       └── plots/
│
└── plots/                              # Differentiable inference output plots
```
