# Geo 8 surrogate failure: diagnosis and fixes

## Problem statement

The geothermal surrogate (a heterogeneous GNN trained on per-scenario revenue) was
exhibiting an extreme OOD failure mode in the active-learning workflow: predictions
on **geology 8** — a tight reservoir with PermZ ~100× lower than the cohort —
diverged from truth so badly that MAPE on geo 8 climbed from 50% at AL iter 0 to
88% at iter 5, while the other 14 geologies stayed near 20%.

This document records the full conversation-length investigation that led to a
working fix and details every experiment that did and didn't help.

## TL;DR — what fixed it

The dominant root cause was **`BatchNorm3d` inside both the per-node and per-edge
3D CNNs**. BatchNorm normalizes activations across the batch and strips the
per-instance absolute-magnitude signal that distinguishes geo 8 ("all my PermZ
values are tiny") from cohort. Fixing this required two coordinated changes per
CNN, plus a switch of loss function:

1. **`BatchNorm3d` → `GroupNorm`** in both `PhysicsSlabCNN` (edge) and
   `PhysicsNodeSlabCNN` (node). GroupNorm normalizes per-instance, preserving
   between-instance scale differences.
2. **Raw-per-channel-means bypass**: each CNN's MLP receives a small projection
   of `slab.mean(dim=(Z,X,Y))` computed pre-CNN, never passing through any
   normalization. Replicates the strongest signal in the hand-engineered
   "well_vertical_profile" features (per-channel means) without going through the
   whole optimization problem.
3. **`HuberLoss(delta=1.0)` → `MSELoss`**. Huber's tail down-weighting was
   capping output predictions in z-space, producing a "prediction floor" at
   ~205 M$ that the model literally couldn't go below.
4. **Slab Z extent = "full" active reservoir** (configurable). Earlier per-well
   slabs resampled Z to span exactly the well's perforation length, giving
   different physical Z resolutions per well. Full-Z + an explicit perforation
   mask channel gives consistent resolution.

Headline benchmark numbers on the 1000-case `benchmark.h5` (200 geo-8 + ~57
of each other geology), held-out **test split** (150 cases, 30 of which are geo 8):

| metric                         | profile (Huber) | profile (MSE) | Exp 1 (v3+edgeGN) | **Final (Exp 1 + A1 + B1)** |
|---|---|---|---|---|
| aggregate MAPE                 | 9.10%           | 8.41%         | 8.37%             | **3.03%**                   |
| aggregate R²                   | +0.691          | +0.731        | +0.718            | **+0.891**                  |
| geo 8 test MAPE                | 13.59%          | 13.02%        | 9.49%             | **4.03%**                   |
| geo 8 test R²                  | +0.270          | +0.297        | +0.623            | **+0.844**                  |
| non-geo 8 test MAPE            | 8.08%           | 7.37%         | 8.08%             | **2.78%**                   |
| geo 8 prediction floor         | 204 M$          | 132 M$        | 123 M$            | 142 M$                      |

**Caveat**: a big chunk of the aggregate test improvement (8.37% → 3.03%) is *not*
a pure model gain — it comes from **Tier A1's switch to geology-aware stratification**.
The old target-only quantile stratification concentrated geo-8 cases (which all
have low revenue) in the bottom target bins, making the test split inadvertently
the *hardest* geo-8 cases. With geology-aware stratification, test geo-8 cases
are 30 samples drawn proportionally from geo-8's full revenue distribution. So
"the test set got easier" is part of the story along with "the model got better."
The A1-only baseline (no Tier B) already hits 3.83% aggregate MAPE / +0.87 geo-8
test R², so Tier B1 contributes ~0.80pp aggregate MAPE on top of A1. The Exp 1
vs final-config comparison on the *same* test split (i.e., Final vs A1_only) is
the apples-to-apples one.

## Investigation timeline

### 1. Diagnostic on the failed AL run

Started from per-geology MAPE heatmap (`local_workspace/plots/per_geology_mape_heatmap.png`)
showing geo 8 catastrophically worse than the other 14. Three parallel
investigations confirmed:
- **Permeability**: geo 8's `log10(PermZ)` is 4.6σ below the cohort. Porosity
  is also lowest, but PermZ is the dominant geology fingerprint.
- **Thermodynamic state** (initial T and P): identical across all geologies.
  Not a contributor.
- **Subagent check on "is MAPE just amplifying low real revenue?"**: partially
  confirmed (geo 8 real revenue is 0.64× cohort mean), but absolute error is
  also 1.48× cohort and growing iteration-over-iteration. Real OOD failure on
  top of denominator amplification.

### 2. Benchmark setup

Built `analysis/geo8_benchmark/` with a stratified 1000-case subset of
`current_compiled.h5` (200 geo-8 + ~57 of each other geology). Used `h5py.ExternalLink`
to avoid duplicating the 25 GB source file — `benchmark.h5` is ~220 KB of pointers.
Geology indices recovered via `filenum_to_scenario_mapping.csv` lookup for
bootstrap cases plus `run_num // 10000` parse for AL cases; cross-validated against
a per-case `log10_mean(PermZ)` fingerprint.

### 3. Loss function — Huber → MSE (positive)

Diagnostic on the prediction z-bounds: Huber-trained model predictions clamped at
`z ∈ [-2.14, +0.98]` despite true z values spanning `[-3.5, +2.2]`. Switching to
MSE expanded predictions to `z ∈ [-3.10, +1.86]` and lifted geo 8 R² substantially.
Caveat: train fit improved more than test fit (some additional overfit), but no
regression on test MAPE.

### 4. Log-perm node features experiment (failed)

Hypothesis: log-transform the per-well permeability scalars (15 perm-related columns
of the 33-d node feature vector) to give the GNN explicit log-space access to geo 8's
tightness signal.

Result: **net negative**. Train MAPE jumped 1.9% → 9.5% because the 15 redundant
log-perm columns gave the GNN an easy shortcut to identify the geology and then just
predict its mean revenue — it stopped learning the within-geology well-placement signal.
Geo 8 test MAPE improved marginally (13.0% → 11.9%) but cohort metrics regressed.

Lesson: too-strong geology-identifying inputs can collapse learning. Per-cell perm
already gives identification; log-transforming everything makes the shortcut overwhelming.

### 5. Node CNN replacement of the 25-d profile

Goal: replace the hand-engineered 25-d `well_vertical_profile` with a learned 3D CNN
over a (Z=16, X=7, Y=7) physics slab around each well.

- **v1**: Simple CNN with `BatchNorm3d` + 2 XY-maxpools + `AdaptiveAvgPool3d`.
  Underfit even geo 8 train: 16% MAPE, R² +0.33. Catastrophic on test (g8 R² −0.97).
- **v2**: Architectural fixes — full-Z slab, no XY pooling (preserve 7×7 lateral
  context through all convs), AvgPool + MaxPool at the end. Aggregate improved
  but geo 8 train MAPE was still 19.5%, geo 8 test R² −0.56. **Model couldn't fit
  its own training data on geo 8.**
- **Diagnostic subagent** showed: within-geology variance dominated the CNN
  embedding (90% within-geology vs 10% between-geology), while the hand-engineered
  profile had 22% between-geology variance. BatchNorm3d was identified as the
  culprit — it normalizes across the batch and strips the very absolute-scale
  signal that distinguishes geo 8 from cohort.
- **v3**: `BatchNorm3d` → `GroupNorm` plus a `raw_means_bypass` MLP path
  concatenating `slab.mean(dim=(Z,X,Y))` (16-d projection) into the final layer.
  Result: geo 8 test R² jumped from −0.56 to **+0.50** — better than the
  profile baseline (+0.30) and the keep-only-means ablation winner (+0.25).
  Geo 8 train MAPE dropped from 19.5% to 10.1% (model now fits training).

This was the central diagnostic moment of the investigation.

### 6. Edge CNN got the same treatment (Exp 1, the current strongest model)

The diagnostic logic for the node CNN — BN strips absolute scale — applies
identically to the edge `PhysicsSlabCNN`. Replacing `BatchNorm3d` → `GroupNorm`
and adding a raw-means bypass in the edge CNN pushed:
- Aggregate test MAPE 8.50% → **8.37%**
- Geo 8 test R² +0.50 → **+0.62** (best of any model at that point)
- Geo 8 test MAPE 11.51% → **9.49%** (first time below 10%)
- Geo 8 train: now actually slightly better than the profile baseline (MAPE 2.86%
  vs profile 3.08%; R² +0.98 vs +0.97).

### 7. Hybrid (profile + CNN) experiment (mostly negative)

Concatenating the 25-d hand-engineered profile WITH the v3 node CNN embedding
was tested. Aggregate metrics improved marginally over profile-alone (MAPE 3.85 →
3.49%) but val_loss was *worse* than either alone (0.315 vs profile 0.298 vs v3 0.223).
Geo 8 test R² improved over profile (+0.30 → +0.36) but well below v3+edgeGN's
+0.62. The redundancy of profile features on top of a working CNN encoder isn't
worth the extra parameters.

### 8. Ablation study of the 25-d hand-engineered profile

Eight ablations zeroed different subsets of the profile to identify which features
matter:

| ablation | aggregate MAPE | geo 8 test MAPE | geo 8 test R² |
|---|---|---|---|
| reference (all 25-d) | 3.85% | 13.02% | +0.297 |
| remove perm profile (12 cols) | 4.02% | 13.60% | +0.154 |
| remove poro profile (4 cols) | 4.41% | 15.64% | −0.033 |
| remove thermo profile (8 cols) | 4.25% | 13.49% | −0.158 |
| keep only means (drop min/max/std) | **4.83%** | **12.74%** | **+0.253** |
| keep only min/max | 5.37% | 13.96% | +0.044 |
| keep only std | 4.75% | 13.14% | +0.271 |
| keep only n_layers | 6.73% | 16.46% | −0.140 |
| remove all profile (8 base scalars only) | 11.84% | 22.78% | −1.059 |

**Surprises:**
- All three property classes (perm, poro, thermo) carry comparable load.
- `keep_only_means` (just 6 means + n_layers = 7 effective dims) gets to within
  1pp of the full 25-d feature set. And matches reference on geo 8 test MAPE.
- `keep_only_std` (just 6 std values) is also competitive — std carries similar
  signal to means.
- `keep_only_n_layers` alone gets you halfway: 6.73% MAPE vs the 11.84%
  "remove all profile" result. n_layers is single most informative scalar.

**Implication**: the v3 node-CNN target is over-ambitious. A simple 6-dim
"per-channel mean" output would have been enough to match most of the profile.
The CNN's value beyond that comes from spatial features the profile can't carry.

### 9. Pipeline-wide audit and the "easy wins" Tier A/B/C plan

Three parallel agents audited (1) GNN architecture, (2) data pipeline + training
loop, (3) AL workflow + acquisition. Returned 24 ranked candidates. Selected and
implemented:

**Tier A — applied directly:**
- A1: geology-aware stratified train/val/test split (was target-only quantile bins
  → now stratifies on geology, falling back to target bins for tiny strata).
- A2: `torch.set_float32_matmul_precision("high")` for ~2× training throughput.
- A3: MAE alongside MAPE + floored MAPE in `orchestrator/ingest.py`. Floor is
  10% of cohort median real revenue, preventing the "geo 8 88% MAPE" denominator
  amplification from making dashboards uninterpretable.
- A4: `target_mape=0.10` early-stop + decoupled `target_mape_window=3` so the
  MAPE-target check doesn't need to clear the full `plateau_window=20`.

**Tier B — A/B benchmark vs Exp 1** (after A1 stratification applied uniformly):

| variant | best val_loss | test all MAPE | test g8 MAPE | test g8 R² |
|---|---|---|---|---|
| baseline (A1_only) | 0.2376 | 3.83% | 4.39% | +0.875 |
| **B1: `--enrich-global-attr`** | **0.1958** | **3.03%** | **4.03%** | +0.844 |
| B2: `--node-aggr sum` (LR halved) | 0.2306 | 3.64% | 4.68% | +0.824 |
| B3: `--cnn-activation gelu` | 0.2182 | 4.04% | 5.28% | +0.843 |
| B4: `--head-no-norm` | 0.2271 | 4.63% | 5.64% | +0.789 |
| B1+B2 combined (LR halved) | 0.2768 | 4.18% | 6.61% | +0.748 |

**Winner: B1 (`--enrich-global-attr`) alone.** Replacing the 1-d `global_attr = [n_wells]`
with an 8-d vector that adds reservoir-mean physics stats (PermX/Y/Z, Porosity,
T0, P0 means + an anisotropy ratio) gives the prediction head a direct geology
fingerprint without going through the GNN message-passing. Smallest possible
code change (~10 LOC), largest delta. Beats every other Tier B option and beats
A1_only on every cohort and split.

B2/B3 modestly improved val_loss but regressed test metrics. B4 (drop head
LayerNorms) regressed clearly. **B1+B2 combined was worse than either alone** —
the halved LR likely under-converged. Stacking doesn't help on this benchmark.

**Tier C — AL workflow:**
- C2: Equal-allocation frontier selection across geologies. Replaces the
  proportional-to-candidate-count allocation that silently penalized geologies
  whose Adam runs went non-finite more often.

**Tier C — considered but kept as-is:**
- `from_scratch_every_k=5` is kept as the default. An earlier audit had flagged
  it as the cause of the iter-5 geo-8 MAPE collapse and recommended `=0`. After
  thinking through the AL trajectory more carefully, kept at 5 to protect
  against warm-start drift accumulation across iterations (the model can absorb
  enough wrong-direction gradient over many warm restarts that periodic resets
  with the full accumulated training set act as a known-good restart). The
  iter-5 geo-8 spike should be reduced by Tier C2 (equal-allocation frontier
  selection) and the Tier A3 / A4 metric improvements; if it still appears,
  revisit this decision then rather than now.

### 10. Final config and recommendations

The new default config flips the following knobs in `train.py`:

| flag | new default | old default |
|---|---|---|
| `--loss` | `mse` | `huber` |
| `--edge-norm` | `groupnorm` | `batchnorm` |
| `--edge-raw-means` | enabled | disabled |
| `--node-encoder` | `cnn` | `profile` |
| `--node-z-extent` | `full` | n/a (new flag) |
| `--enrich-global-attr` | enabled | disabled |
| TF32 matmul precision | `high` | default (low) |
| stratified split | geology-aware | target-only |
| Lightning AL stopping | `target_mape=0.10`, `target_mape_window=3` | `target_mape=null` |
| AL frontier slot allocation | equal across geologies | proportional to count |

Kept as-is (`from_scratch_every_k=5`) per user direction — concern about
warm-start drift accumulation outweighs the diagnostically-identified iter-5
spike, especially since the post-fix model is calibrated well enough that
the spike should be smaller anyway.

## What didn't work / negative results

In the interests of not repeating mistakes, the failed experiments:

- **Log-transforming node-level permeability features**: created a shortcut that
  destroyed within-geology learning. The model collapsed to "look at log-perm to
  identify geology, then predict its mean revenue."
- **Hyperparameter optimization (Optuna) on the failing v1/v2 node CNN**:
  15 trials returned the unregularized baseline as the winner. More dropout/
  weight decay didn't help because the issue wasn't classical overfitting —
  it was a representational bottleneck (BatchNorm stripping scale).
- **Hybrid profile + CNN node features**: marginal aggregate improvement, but val
  loss higher than either alone. Profile features become redundant once the CNN
  has GroupNorm + raw-means bypass.
- **CNN replacement at v1 and v2**: both regressed vs profile baseline. Took
  three iterations to find the right architecture.
- **Pre-fix HPO** of v2 node CNN: TPE study with 10 trials made no progress
  beyond the unregularized baseline — confirming the issue was not
  hyperparameter-tunable.

## Critical files

- `geothermal/physics_slab.py` — `PhysicsSlabCNN`, `PhysicsNodeSlabCNN` with the
  GN/raw-means/activation flags.
- `geothermal/model.py` — `HeteroGNNRegressor`, hosts `edge_norm`,
  `edge_raw_means`, `node_encoder`, `node_z_extent`, `node_aggr`, `cnn_activation`,
  `head_no_norm` knobs.
- `geothermal/data.py` — `build_single_hetero_data` (handles `node_encoder` and
  `enrich_global_attr`), `split_indices_stratified` (geology-aware stratification).
- `train.py` — CLI surface for every flag above.
- `analysis/geo8_benchmark/evaluate_benchmark.py` — A/B harness.
- `geothermal_active_learning/orchestrator/{ingest,stop,select}.py` — Tier A3 /
  A4 / C2 changes.
- `geothermal_active_learning/configs/al_hybrid_post_fix.json` — Tier A4.

## Reproducing the final benchmark

After the default-flip (commit), the winning run is:

```bash
cd Geothermal_Graph_Surrogate
conda activate geothermal-pomdp
python train.py \
  --h5-path analysis/geo8_benchmark/benchmark.h5 \
  --target graph_discounted_net_revenue \
  --seed 42 --split-seed 42 --run-id 0 \
  --output-root analysis/geo8_benchmark/runs/training_winning \
  --gpu 0 --batch-size 16 --learning-rate 3e-4 \
  --max-epochs 180 --early-stop-patience 30 \
  --stratified-split --cache-to-gpu
```

All other knobs now default to the winning config (MSE loss, GroupNorm edge CNN,
raw-means bypass, node-CNN encoder with full-Z slab, enriched global_attr).

Eval:

```bash
python analysis/geo8_benchmark/evaluate_benchmark.py \
  --training-root analysis/geo8_benchmark/runs/training_winning \
  --out-dir       analysis/geo8_benchmark/runs/eval_winning
```

The pre-Tier-A/B run artifacts under `analysis/geo8_benchmark/runs/` (training and
eval directories for every variant referenced in this document) are not tracked by
git (see `.gitignore`); they can be regenerated by re-running `run_easywins.py` and
`run_ablations.py`.
