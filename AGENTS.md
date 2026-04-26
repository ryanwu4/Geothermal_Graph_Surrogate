# AGENTS

## Purpose
Use this file to become productive quickly in this repository.

## Critical Environment Rule
- Always use the conda environment `geothermal-pomdp` for any Python command in this repo.
- Environment definition: [environment.yml](environment.yml)
- Existing always-on rule: [.agents/rules/python-env.md](.agents/rules/python-env.md)

Recommended setup:
```bash
conda env create -f environment.yml
conda activate geothermal-pomdp
```

## Start Here
- Project overview and model/data details: [README.md](README.md)
- Training entry point: [train.py](train.py)
- Data preprocessing: [preprocess_h5.py](preprocess_h5.py)
- H5 compilation helper: [compile_minimal_geothermal_h5.py](compile_minimal_geothermal_h5.py)
- Discounted revenue objective notes: [NPV_OBJECTIVE_DIFFERENCES.md](NPV_OBJECTIVE_DIFFERENCES.md)

## Common Commands
```bash
# Activate environment first
conda activate geothermal-pomdp

# Train model
python train.py --h5-path data_test/minimal_compiled_tp.h5

# Preprocess raw simulation H5 files
python preprocess_h5.py --help

# Build minimal compiled H5 from simulation outputs
python compile_minimal_geothermal_h5.py --help

```

## Project Map
- Core package: [geothermal/](geothermal)
- Model definition: [geothermal/model.py](geothermal/model.py)
- Physics slab encoders and 3D cropping: [geothermal/physics_slab.py](geothermal/physics_slab.py)
- Data loading/scaling/ablation logic: [geothermal/data.py](geothermal/data.py)
- Evaluation and plots: [geothermal/evaluation.py](geothermal/evaluation.py)
- Configs: [configs/](configs)
- Example datasets: [data_test/](data_test)
- Training outputs/checkpoints: [trained/](trained), [lightning_logs/](lightning_logs)

## Repo-Specific Conventions
- `train.py` expects compiled H5 inputs, not raw simulator H5 files.
- Use consistent normalization artifacts between training and inference (see README pipeline sections).
- `graph_discounted_net_revenue` workflows depend on economics config files under [configs/](configs).

## Agent Behavior Notes
- Prefer linking to existing docs over duplicating long explanations.
- Keep edits minimal and focused.
- If a command may be expensive, start with `--help` or a small dataset in [data_test/](data_test).
