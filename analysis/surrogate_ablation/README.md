# Surrogate-architecture ablation study

YAML-driven training matrix comparing architecture variants across the three
benchmark datasets (`ablation_datasets/`, see its README) and three objectives
(`graph_energy_total`, `graph_discounted_net_revenue`, `node_wept_final`).

## Variants

| name | what it tests | mechanism |
|---|---|---|
| `gnn_default` | production architecture | train.py, `--edge-encoder cnn` |
| `gnn_dist_edge` | edge geology-CNN contribution | `--edge-encoder dist` (edges = inter-well distance only) |
| `gnn_type_only_nodes` | node feature contribution | `--node-features type_only --node-encoder profile` (nodes = is_injector flag) |
| `gnn_svd_edge` | learned CNN vs frozen linear basis | `--edge-encoder svd`, basis auto-fit per dataset on the train split (fit_svd.py) |
| `mlp` | everything (floor baseline) | train_baseline.py: well xyz+type + geology one-hot → MLP |
| `global_cnn` | graph structure vs monolithic CNN | train_baseline.py: full-grid 3D CNN + well-marker heatmap channels |

## Usage

```bash
cd /home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate
PY=/home/rwu4/miniconda3/envs/geothermal-pomdp/bin/python

$PY analysis/surrogate_ablation/run_matrix.py --config analysis/surrogate_ablation/configs/stage1.yaml --dry-run
$PY analysis/surrogate_ablation/run_matrix.py --config analysis/surrogate_ablation/configs/stage1.yaml
# subset:            ... --only variant=mlp,dataset=simple_2pair
# collate only:      ... --collate-only
```

- **GPU choice**: `gpu:` per variant in the YAML; per-cell via `overrides`
  (matched on any subset of variant/dataset/target/seed). The runner sets
  `CUDA_VISIBLE_DEVICES` per job — never edit `--gpu`.
- **Seed top-up**: edit `seeds: [42]` → `[42, 43, 44]` and re-run; completed
  jobs (parseable `metrics_summary.json`) are skipped.
- **Fairness**: one `split_seed` for every job → byte-identical geology-
  stratified splits across variants AND targets (collation asserts this from
  the test_predictions.csv files). Baselines reuse the same loader, split
  function, and target scaler as train.py.
- **Retrain a cell**: delete its job dir under `runs/<stage>/...` and re-run.
- Results: `runs/<stage>/summary.csv` + pivot table printed at the end;
  per-run artifacts in `runs/<stage>/<dataset>/<target>/<variant>/seed<N>/`
  (`train.log`, `job.json`, checkpoints, plots).

## Extra axes (stages 2/3)

- `train_sizes: [256, 512]` (config key) → `--train-subsample N`: geology-
  proportional subsample of the train split, seeded by the run seed (same
  subset across variants at the same seed; varies across seeds). Val/test
  splits untouched → metrics comparable with full-size runs.
- `holdout_geologies: [8, ...]` → `--holdout-geologies G`: geology G excluded
  from train/val/test; its cases become the `test_ood` split. The MLP's
  one-hot column for G is never active in training — by design (it cannot
  generalize to unseen geologies; grid-reading models can).
- The SVD pre-step fits one basis per (dataset, split params, size, seed,
  holdout) combination; split-shaping flags are forbidden in variant/override
  `flags:` (runner hard-errors) so bases can never desync from jobs.
- New variant `gnn_no_cnn` = `{edge-encoder: dist, node-features: type_only,
  node-encoder: profile}`: no CNN information to nodes OR edges; geology
  reaches the model only through the 8-d global-attr stats.

## Interpretation caveats

- **cross_eval.csv metrics are NOT comparable to summary.csv's `mape`**: the
  stage CSVs use evaluation.py's significance-masked, 99th-pct-clamped MAPE;
  cross_eval reports wMAPE/median-APE. Only R²/Spearman (and case sets) are
  directly cross-comparable.
- **Physics normalization**: per-channel min/max constants are fixed a priori
  and byte-identical across all three datasets (verified); they are never
  refit per split — static preprocessing, not leakage.
- Single-seed margins below rerun jitter (~2e-3 relative on predictions;
  TF32, deterministic=False) should not be over-read — use the seed top-ups.

- `gnn_type_only_nodes` is a *node-side geology removed* ablation, not
  geology-blind: it drops the 9 node scalars AND the node-slab CNN jointly
  (intentional pairing) while keeping edge slabs and the 8-d enriched global
  attr. Attribute results to the joint change.
- The `mlp` baseline receives an explicit 15-way geology one-hot — a cleaner
  geology-identity signal than the GNN's physics-stat global fingerprint;
  fine for a floor baseline, remember when reading its numbers.
- `split_indices_stratified` has a target-dependent random-split escape hatch
  when a target has <3 unique quantile edges (near-constant target). Not
  triggerable on these datasets/targets, but if a future target is added,
  re-verify the collation split-identity assertion passes.
- `type_only` checkpoints record `node_features_mode` in hparams
  (hparams_to_data_kwargs threads it); older analysis code that hand-builds
  graphs must pass `node_features_mode="type_only"`.

## Results figures (figures/, regenerate with plot_final.py)

- `final_architecture_r2.png` — full-data test R² per variant × dataset × objective (all 7 variants).
- `final_data_efficiency.png` — test R² vs train size {256, 512, full}, revenue + WEPT.
- `final_holdout_geo.png` — OOD test R² on held-out geologies (geo 3 typical, geo 8 tight).

Headline: `gnn_type_only_nodes` matches/beats the default everywhere (node scalars
redundant); `gnn_no_cnn` is decisively worst, esp. per-well WEPT (physics-slab CNN
is load-bearing); grid-reading models degrade far less than the MLP on the unseen
hard geology.

## Production safety

train.py/data.py/model.py changes for variants 3–4 are additive with unchanged
defaults (asserted against the AL-contract flag set). Baselines live entirely
in this directory; their checkpoints are never read by the AL pipeline.
