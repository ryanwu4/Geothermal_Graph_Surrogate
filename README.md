# Geothermal GNN — Network Architecture & Usage Guide

## Overview

This project uses a **heterogeneous graph neural network (Hetero-GNN)** to predict field-level energy production from geothermal well configurations. Wells are modeled as nodes in a graph, with edges representing subsurface geological connectivity computed via **A\* pathfinding** through 3D permeability grids. The model is built with [PyTorch Geometric](https://pyg.org) and trained via [PyTorch Lightning](https://lightning.ai).

---

## Network Architecture

### Graph Structure

Each simulation run is encoded as a single **heterogeneous graph**:

| Component | Description |
|---|---|
| **Nodes** | One node per well (injector or extractor) |
| **Edge types** | 4 typed relations: `inj→inj`, `ext→ext`, `inj→ext`, `ext→inj` |
| **Edge source** | A\* paths through 3D geology grids + k-nearest-neighbor fallback |

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

### Edge Features (14 dimensions per edge)

Each edge carries attributes computed along the A\* path between two wells:

| Index | Feature |
|---|---|
| 0 | Path length (grid cells) |
| 1 | Time-of-flight cost (porosity/permeability integrated) |
| 2–4 | Min, max, harmonic-mean permeability along path |
| 5–7 | Min, max, harmonic-mean porosity along path |
| 8–9 | Temperature and pressure deltas (endpoint to endpoint) |
| 10–11 | Temperature and pressure gradients (delta / path length) |
| 12–13 | Min temperature and min pressure along path |

### Global Features (1 dimension)

| Index | Feature |
|---|---|
| 0 | Well count (number of nodes in the graph) |

### Model Architecture (`HeteroGNNRegressor`)

```
Input: HeteroData graph
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
- **NNConv** with edge-conditioned message passing: a 2-layer MLP transforms the 14-dim edge features into a weight matrix for each edge
- **Heterogeneous convolutions** (`HeteroConv`) apply separate NNConv operators per edge type, then aggregate via summation
- **Triple pooling** (sum + mean + max) captures both magnitude and distribution information across the well network
- **GELU** activation throughout (smoother than ReLU)
- **LayerNorm** per GNN layer for stable training

### Default Hyperparameters

| Parameter | Default |
|---|---|
| Hidden dim | 32 |
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

### Step 1: Compile Raw HDF5 → Minimal Dataset

```bash
python compile_minimal_geothermal_h5.py \
    --input-dir data/ \
    --output-file data/compiled.h5
```

This processes each raw simulation HDF5 file:
1. Extracts well locations, injection rates, and reservoir properties
2. Computes vertical profile statistics (25 features per well)
3. Runs **A\* pathfinding** through the 3D permeability grid to compute geology-aware edges
4. Extracts output targets (energy production, WEPT timeseries, T/P profiles)
5. Writes a single compiled HDF5 with all simulations indexed by case ID

### Step 2: Train

```bash
python train.py \
    --h5-path data/compiled.h5 \
    --target graph_energy_total \
    --max-epochs 180 \
    --batch-size 16 \
    --stratified-split
```

**With top-k% withholding** (removes top 10% of runs by energy production):
```bash
python train.py \
    --h5-path data/compiled.h5 \
    --target graph_energy_total \
    --withhold-top-pct 10
```

Training produces:
- Best model checkpoint: `lightning_logs/.../checkpoints/best-*.ckpt`
- Fitted scaler: `lightning_logs/.../checkpoints/scaler.pkl`
- Evaluation plots: scatter, error distributions, loss curves, extreme-error graphs
- (If withholding): `withheld_runs.json` + `withholding_histogram.png`

### Step 3: Infer

```bash
python infer.py \
    --h5-path data/new_runs.h5 \
    --checkpoint lightning_logs/.../checkpoints/best-*.ckpt \
    --scaler-path lightning_logs/.../checkpoints/scaler.pkl \
    --output predictions.csv
```

Output CSV columns: `case_id`, `predicted_total_energy`, `actual_total_energy`

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
python train.py --h5-path data/compiled.h5 --ablate vertical_profile edge_thermo
```

| Group | Type | Columns zeroed |
|---|---|---|
| `vertical_profile` | Node | 8–32 (vertical profile stats) |
| `base_perm` | Node | 2–4 (perm x/y/z) |
| `base_thermo` | Node | 6–7 (temp0, press0) |
| `edge_perm` | Edge | 2–4 (min/max/hm perm) |
| `edge_poro` | Edge | 5–7 (min/max/hm poro) |
| `edge_thermo` | Edge | 8–13 (T/P deltas, grads, mins) |
| `edge_all` | Edge | 0–13 (all edge features) |

### Ablation Study Results

Baseline: all 33 node features + 14 edge features, `dropout=0.0`, `hidden=32`, `layers=2`, target = `graph_energy_total`.

| Removed Features | Best Val Loss | Val MAE (×10¹²) | Δ vs Baseline |
|---|:---:|:---:|:---:|
| *(none — baseline)* | 0.1452 | 6.48 | — |
| base_perm + edge_perm (6 cols) | 0.1276 | 6.20 | 🟢 −12% |
| edge_perm (3 edge) | 0.1510 | 6.79 | 🟡 +4% |
| edge_all (14 edge) | 0.1767 | 7.47 | 🔴 +22% |
| edge_thermo (6 edge) | 0.1814 | 7.68 | 🔴 +25% |
| vertical_profile (25 node) | 0.2293 | 8.53 | 🔴 +58% |

### Key Findings

1. **`params_scalar` was actively hurting generalization** — removing the 26 scenario parameters (previously tiled identically across all nodes) dropped val loss by 39%. The model memorized these as graph-level fingerprints rather than learning transferable physics. This motivated its permanent removal from the codebase.

2. **`vertical_profile` is the most important feature group** — removing it causes +58% val loss. The vertical property summary statistics capture critical reservoir heterogeneity that single-point bottom-of-well properties miss.

3. **Edge thermodynamics are highly informative** — inter-well T/P deltas and gradients contribute +25% val loss when removed, consistent with Darcy's law (pressure gradients drive flow, thermal gradients drive heat transfer).

4. **Permeability is partially redundant between nodes and edges** — removing `edge_perm` alone barely hurts (+4%), but removing both `base_perm` and `edge_perm` together actually *helps* (−12%), suggesting conflicting permeability signals at different scales add noise.

5. **Pure graph topology has value, but edge physics matter** — removing all edge features (`edge_all`) hurts by +22%, confirming A\* path attributes contribute beyond connectivity structure alone.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate geothermal_min
```

---

## Project Structure

```
├── environment.yml                     # Conda environment spec
├── train.py                            # Training entry point
├── infer.py                            # Inference entry point
├── compile_minimal_geothermal_h5.py    # Raw HDF5 → compiled dataset
├── build_geology_graph.py              # Backward-compat wrapper
│
├── geothermal/                         # Core package
│   ├── __init__.py
│   ├── model.py                        # HeteroGNNRegressor + constants
│   ├── data.py                         # Loading, scaling, splitting, withholding
│   ├── evaluation.py                   # Metrics, plotting, error analysis
│   └── geology_graph.py               # A* search + edge generation
│
└── lightning_logs/                     # Training outputs (checkpoints, plots)
```
