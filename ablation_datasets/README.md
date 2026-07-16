# Ablation benchmark datasets

Three IX-labeled datasets for the surrogate-architecture ablation study
(GNN+CNN vs. MLP / SVD edge embeddings / no-edge / no-node / whole-domain-CNN
baselines). All three share **byte-identical physics normalization**
(`norm_config`) and cover the same 15-geology ensemble (scenarios 71–85), so
models are comparable across datasets and cross-dataset evaluation is valid.

| Dataset | Cases | Wells/case | Depth (requested) | Placement | Source |
|---|---|---|---|---|---|
| `simple_2pair/` | 2048 | 4 (2 inj + 2 prod) | 50–55 | LHS xy + LHS depth | seed-only batch, 2026-07-08 |
| `hard_fulldepth/` | 2047 | 12 (6 inj + 6 prod) | 11–70 | LHS xy + LHS depth | seed-only batch, 2026-07-09 |
| `highperf_cma/` | 3287 | 12 (6 inj + 6 prod) | 50–55 | CMA-multistart elites (92%) + LHS seed (8%) | gated_seed42 AL run (hardlink to its `current_compiled.h5`) |

Per-dataset sidecars: `*norm_config.json` (physics min/max), `case_geology_map.json`
(case → geology index, used by geology-stratified splits). `highperf_cma/` adds
`ablation_highperf_case_composition.json` (seed vs. CMA case lists for
analysis-time subsetting).

## Provenance / reproduction

- Generator configs: `geothermal_active_learning/configs/ablation_seed_{simple_2pair,hard_fulldepth}.json`
  (driver: `scripts/run_seed_lhs.py`; sampling is deterministic, seed 42, even
  split across geologies: 137×8 + 136×7).
- Master manifest: `geothermal_active_learning/configs/ablation_datasets.json`.
- All compiled post the 2026-06-28 perforation/perf_span fix.

## Objectives (train.py --target)

`graph_energy_total`, `graph_discounted_net_revenue` (revenue only — no
CAPEX/OPEX), `node_wept_final` (per-producer WEPT at year 30).

## Known dataset properties

- **Realized vs. requested depth**: compiled `wells.depth` is the deepest
  *active* perforated cell — columns whose rock ends shallower than the
  requested depth perforate shallower (SIMPLE realized 32–55; ~35% of wells
  < 50). Same behavior in every prior dataset.
- HARD's depth cap is per-geology `min(70, z_cutoff−1)`; z_cutoff is 63 here,
  so realized depths span up to 62.
- `highperf_cma/` is optimizer-biased by construction (exploit picks around
  CMA optima) and contains ~8% LHS seed cases — subset via the composition
  sidecar if a pure-CMA distribution is needed.
- The 4-well graphs in `simple_2pair/` have 1 same-type KNN neighbor per well
  (vs. 2 in the 12-well datasets) — a property of the k=2-per-type topology on
  2 pairs, not a bug.
- HARD originally compiled 2048 cases; one case
  (`..._83_run120057_iter0000`, two producers LHS-projected onto the same
  grid column → 11 discovered wells) was **removed 2026-07-09** so all
  architecture variants train on identical fixed-width data. HARD = 2047.
- `highperf_cma/case_geology_map.json` was regenerated 2026-07-09 to cover
  all 3287 cases (original 256-entry sidecar backed up as `.bak_256`);
  generator: `analysis/surrogate_ablation/build_geology_map.py`.

## Figures

`figures/well_placement_coverage.png` — spatial xy density, realized depth
distributions, per-geology case counts.
`figures/objective_coverage.png` — distributions of the three objectives per
dataset. Regenerate with the scripts referenced in `figures/` (extraction NPZ
alongside).
