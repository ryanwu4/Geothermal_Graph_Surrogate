# Geothermal GNN — Network Architecture & Usage Guide

## Overview

This project uses a **heterogeneous graph neural network (Hetero-GNN)** to predict field-level energy production from geothermal well configurations. Wells are modeled as nodes in a graph, with edges representing subsurface geological connectivity via **k-nearest neighbors**. Edge features are dynamically extracted by a **3D CNN** that crops and processes local geological volumes between each well pair. The model is built with [PyTorch Geometric](https://pyg.org) and trained via [PyTorch Lightning](https://lightning.ai).

The entire pipeline — from continuous well coordinates through physics interpolation, CNN feature extraction, and GNN prediction — is **end-to-end differentiable**, enabling gradient-based optimization of well placement.

---

## Network Architecture

### Graph Structure

Each simulation run is encoded as a single **heterogeneous graph**:

| Component | Description |
|---|---|
| **Nodes** | One node per well (injector or producer) |
| **Edge types** | 4 typed relations: `inj→inj`, `ext→ext`, `inj→ext`, `ext→inj` |
| **Edge source** | K-nearest neighbors (k=2 per type) based on 3D Euclidean distance |

### Node Features (33 dimensions)

| Index | Feature | Source |
|---|---|---|
| 0 | Injection rate | Well operating parameter |
| 1 | Depth (deepest perforated layer) | Well geometry |
| 2–4 | Permeability (x, y, z) | Reservoir property at well |
| 5 | Porosity | Reservoir property at well |
| 6 | Initial temperature | Reservoir state |
| 7 | Initial pressure | Reservoir state |
| 8–32 | Vertical profile statistics (25 features) | Mean/min/max/std over 6 properties × 4 stats + n_layers |

### Edge Features (dynamically computed by 3D CNN)

Edge features are **not hardcoded**. Instead, a `PhysicsSlabCNN` dynamically extracts them:

1. For each well pair (edge), a **3D sub-volume** of the geology grid is cropped using differentiable `grid_sample` interpolation
2. The crop is resized to a fixed 32×32×16 volume and concatenated with 3D Gaussian heatmaps marking each well's position
3. A **3D CNN** (Conv3d → BatchNorm → GELU → MaxPool) processes the volume into a latent vector
4. The latent vector is concatenated with the Euclidean distance Δs between the wells
5. An MLP compresses the result to the configured `latent_edge_dim` (default: 32)

**Physics channels** processed by the CNN:
- PermX, PermY, PermZ (log-transformed, min-max normalized)
- Porosity, Temperature0, Pressure0 (min-max normalized)
- valid_mask (binary rock/void indicator)

### Global Features (1 dimension)

| Index | Feature |
|---|---|
| 0 | Well count (number of nodes in the graph) |

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

| Parameter | Default |
|---|---|
| Hidden dim | 32 |
| Latent edge dim | 32 |
| CNN channels | 8 → 16 → 32 → 32 |
| Num GNN layers | 2 |
| Dropout | 0.0 |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Loss | Huber (δ=1.0) |
| Optimizer | AdamW |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Early stopping | patience=60 on val_loss |

---

## Data Pipeline

### Step 1: Preprocess Raw HDF5 → Normalized Dataset

```bash
python preprocess_h5.py \
    --input-dir data/ \
    --output-h5 compiled_full_CNN.h5 \
    --norm-config norm_config.json
```

This processes each raw simulation HDF5 file:
1. Extracts well locations, injection rates, and reservoir properties
2. Computes vertical profile statistics (25 features per well)
3. Identifies valid rock cells and computes Z-axis depth cutoff (first layer where ≥95% of cells are inactive)
4. Log-transforms permeability, then min-max normalizes all physics channels to [0, 1]
5. Saves the processed 3D physics tensors and well tables into a compiled HDF5

**First run**: computes global normalization statistics across all files and writes `norm_config.json`.
**Subsequent runs**: reuses `norm_config.json` for consistent normalization.

Use `--compute-only` to only compute normalization stats without building the H5.

### Step 2: Train

```bash
python train.py \
    --h5-path compiled_full_CNN.h5 \
    --target graph_energy_total \
    --max-epochs 180 \
    --batch-size 16 \
    --stratified-split \
    --gpu 0
```

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
4. Optimizes well (X, Y, Z) coordinates via Adam gradient ascent to maximize predicted energy
5. **Feasible direction projection**: gradients pointing outside the grid domain are projected along the boundary; Adam momentum is flushed for boundary-violating components
6. Outputs a 2D trajectory plot (wells over log-PermX background) alongside an energy-vs-iteration curve to `plots/`

All plots are saved with timestamps (`plots/well_trajectories_2D_Energy_YYYYMMDD_HHMMSS.png`) to prevent overwriting.

---

## Prediction Targets

| Target flag | Level | Output dim | Description |
|---|---|---|---|
| `graph_energy_total` | Graph | 1 | Total field energy production (cumulative, final timestep) |
| `graph_energy_rate` | Graph | 1 | Field energy production rate (first timestep) |
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
