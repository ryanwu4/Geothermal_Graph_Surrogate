#!/usr/bin/env python3
"""YAML-driven matrix runner for the surrogate-architecture ablation study.

Expands variants x datasets x targets x seeds into jobs, schedules them onto
GPUs (per-variant/per-run GPU pins, max-concurrent-per-GPU, optional
per-dataset concurrency guards for RAM/VRAM), skips completed runs (so topping
up seeds = editing the YAML and re-running), auto-fits per-dataset SVD bases
when an svd variant is pending, and collates results.

Usage:
    python run_matrix.py --config configs/stage1.yaml [--dry-run]
        [--collate-only] [--only variant=mlp,dataset=simple_2pair]

GPU isolation follows the geo8_benchmark precedent: each subprocess gets
CUDA_VISIBLE_DEVICES=<physical gpu> and --gpu 0 (never a multi-index --gpu:
--cache-to-gpu builds torch.device(f"cuda:{gpu}") and would crash).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

SHARED_FLAG_MAP = {
    # shared-config key -> CLI flag (value flags)
    "split_seed": "--split-seed",
    "val_fraction": "--val-fraction",
    "test_fraction": "--test-fraction",
    "batch_size": "--batch-size",
    "max_epochs": "--max-epochs",
    "early_stop_patience": "--early-stop-patience",
    "max_cases": "--max-cases",
}
LOGGER_NAME = {"train_py": "geothermal_hetero_gnn", "baseline": "baseline"}


@dataclass
class Job:
    variant: str
    dataset: str
    target: str
    seed: int
    kind: str
    gpu: int
    flags: dict = field(default_factory=dict)
    train_size: int | None = None      # --train-subsample axis
    holdout_geo: int | None = None     # --holdout-geologies axis

    @property
    def key(self) -> dict:
        return {"variant": self.variant, "dataset": self.dataset,
                "target": self.target, "seed": self.seed,
                "train_size": self.train_size, "holdout_geo": self.holdout_geo}

    @property
    def name(self) -> str:
        suffix = f"seed{self.seed}"
        if self.train_size is not None:
            suffix += f"_n{self.train_size}"
        if self.holdout_geo is not None:
            suffix += f"_geo{self.holdout_geo}"
        return suffix

    def dir(self, output_root: Path) -> Path:
        return output_root / self.dataset / self.target / self.variant / self.name

    def metrics_path(self, output_root: Path) -> Path:
        return (self.dir(output_root) / LOGGER_NAME[self.kind] / "run_00"
                / "plots" / "metrics_summary.json")

    def done(self, output_root: Path, expected_cmd: list[str] | None = None) -> bool:
        """Complete iff metrics_summary.json parses AND (when expected_cmd is
        given) the run was produced by the same command — so editing the config
        marks affected cells stale instead of silently mixing configs."""
        p = self.metrics_path(output_root)
        if not p.exists():
            return False
        try:
            json.loads(p.read_text())
        except Exception:
            return False
        if expected_cmd is not None:
            try:
                meta = json.loads((self.dir(output_root) / "job.json").read_text())
            except Exception:
                return False
            if meta.get("cmd") != expected_cmd:
                print(f"[matrix] STALE (config changed) "
                      f"{self.variant}/{self.dataset}/{self.target}/seed{self.seed}")
                return False
        return True


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("shared", {})
    cfg.setdefault("overrides", [])
    cfg.setdefault("exclude", [])
    cfg.setdefault("scheduler", {})
    return cfg


def match(rule: dict, job: Job) -> bool:
    return all(str(job.key.get(k)) == str(v) for k, v in rule.items())


def expand_jobs(cfg: dict) -> list[Job]:
    jobs: list[Job] = []
    gpus = cfg["scheduler"].get("gpus", [0])
    train_sizes = cfg.get("train_sizes") or [None]
    holdout_geos = cfg.get("holdout_geologies") or [None]
    i = 0
    for vname, vspec in cfg["variants"].items():
        for dname in cfg["datasets"]:
            for target in cfg["targets"]:
                for seed in cfg["seeds"]:
                    for size in train_sizes:
                        for hgeo in holdout_geos:
                            job = Job(
                                variant=vname, dataset=dname, target=target,
                                seed=int(seed), kind=vspec["kind"],
                                gpu=int(vspec.get("gpu", gpus[i % len(gpus)])),
                                flags=dict(vspec.get("flags", {})),
                                train_size=None if size is None else int(size),
                                holdout_geo=None if hgeo is None else int(hgeo),
                            )
                            i += 1
                            if any(match(r, job) for r in cfg["exclude"]):
                                continue
                            for ov in cfg["overrides"]:
                                if match(ov.get("match", {}), job):
                                    if "gpu" in ov:
                                        job.gpu = int(ov["gpu"])
                                    job.flags.update(ov.get("flags", {}))
                            jobs.append(job)
    return jobs


def build_command(cfg: dict, job: Job, output_root: Path,
                  svd_paths: dict[str, Path]) -> list[str]:
    py = cfg["python"]
    ds = cfg["datasets"][job.dataset]
    entry = "train.py" if job.kind == "train_py" else str(HERE / "train_baseline.py")
    cmd = [
        py, entry,
        "--h5-path", str(REPO / ds["h5_path"]),
        "--target", job.target,
        "--seed", str(job.seed),
        "--run-id", "0",
        "--output-root", str(job.dir(output_root)),
        "--gpu", "0",
    ]
    shared = _merged_shared(cfg, job.dataset)
    for key, flag in SHARED_FLAG_MAP.items():
        if shared.get(key) is not None:
            cmd += [flag, str(shared[key])]
    if shared.get("cache_to_gpu", False):
        cmd.append("--cache-to-gpu")
    if job.train_size is not None:
        cmd += ["--train-subsample", str(job.train_size)]
    if job.holdout_geo is not None:
        cmd += ["--holdout-geologies", str(job.holdout_geo)]
    for k, v in job.flags.items():
        if v is None or v is True or v == "":
            cmd.append(f"--{k}")           # valueless / store_true flag
        else:
            cmd += [f"--{k}", str(v)]
    if job.kind == "train_py":
        cmd.append("--require-geology-map")
        if job.flags.get("edge-encoder") == "svd":
            cmd += ["--svd-weights-path", str(svd_paths[_svd_key(cfg, job)])]
    return cmd


# Flags that shape the split/scaler; forbidden in variant/override flags —
# they must come through `shared` or the train_sizes/holdout_geologies axes,
# else the SVD basis and split-identity assertions silently desync (audit F4).
_SPLIT_SHAPING_FLAGS = {
    "split-seed", "val-fraction", "test-fraction", "max-cases",
    "train-subsample", "holdout-geologies", "stratified-split",
    "no-stratified-split", "withhold-top-pct",
}


def _merged_shared(cfg: dict, dname: str) -> dict:
    ds = cfg["datasets"][dname]
    return {**cfg["shared"], **{k: v for k, v in ds.items()
                                if k not in ("h5_path", "max_concurrent")}}


def _svd_key(cfg: dict, job: Job) -> tuple:
    """Everything that changes the train slabs the basis must be fit on."""
    sh = _merged_shared(cfg, job.dataset)
    return (
        job.dataset,
        sh.get("split_seed", 42), sh.get("val_fraction", 0.15),
        sh.get("test_fraction", 0.15), sh.get("max_cases"),
        job.train_size, job.seed if job.train_size is not None else None,
        job.holdout_geo,
    )


def _svd_path(output_root: Path, key: tuple, k: int) -> Path:
    dname, ss, vf, tf, mc, size, sseed, hgeo = key
    tag = f"k{k}_split{ss}_v{vf}_t{tf}"
    if mc:
        tag += f"_mc{mc}"
    if size is not None:
        tag += f"_n{size}_ss{sseed}"
    if hgeo is not None:
        tag += f"_hg{hgeo}"
    return output_root / "svd" / f"{dname}_{tag}.pt"


def ensure_svd(cfg: dict, jobs: list[Job], output_root: Path,
               dry_run: bool) -> dict[tuple, Path]:
    svd_cfg = cfg.get("svd", {})
    k = int(svd_cfg.get("k", 32))

    for j in jobs:
        bad = _SPLIT_SHAPING_FLAGS & set(j.flags)
        if bad:
            raise SystemExit(
                f"split-shaping flag(s) {sorted(bad)} on variant "
                f"{j.variant!r} — move them to `shared`/axes so the SVD "
                f"basis and split assertions stay consistent."
            )

    paths: dict[tuple, Path] = {}
    fit_jobs: dict[tuple, Job] = {}
    for j in jobs:
        key = _svd_key(cfg, j)
        paths[key] = _svd_path(output_root, key, k)
        if (j.kind == "train_py" and j.flags.get("edge-encoder") == "svd"
                and key not in fit_jobs
                and not j.done(output_root, build_command(cfg, j, output_root, paths))):
            fit_jobs[key] = j

    for key, j in sorted(fit_jobs.items(), key=lambda kv: str(kv[0])):
        out = paths[key]
        if out.exists():
            continue
        dname, ss, vf, tf, mc, size, sseed, hgeo = key
        cmd = [
            cfg["python"], str(HERE / "fit_svd.py"),
            "--h5-path", str(REPO / cfg["datasets"][dname]["h5_path"]),
            "--output-path", str(out),
            "--k", str(k),
            "--split-seed", str(ss),
            "--val-fraction", str(vf),
            "--test-fraction", str(tf),
            "--max-cases-fit", str(svd_cfg.get("max_cases_fit", 300)),
            "--max-slabs", str(svd_cfg.get("max_slabs", 12000)),
        ]
        if mc:
            cmd += ["--max-cases", str(mc)]
        if size is not None:
            cmd += ["--train-subsample", str(size), "--subsample-seed", str(sseed)]
        if hgeo is not None:
            cmd += ["--holdout-geologies", str(hgeo)]
        if dry_run:
            print(f"[svd] WOULD fit: {shlex.join(cmd)}")
            continue
        print(f"[svd] fitting basis {out.name}")
        out.parent.mkdir(parents=True, exist_ok=True)
        log = out.with_suffix(".log")
        with open(log, "w") as f:
            rc = subprocess.run(cmd, cwd=str(REPO), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"fit_svd failed for {out.name}; see {log}")
    return paths


def run_jobs(cfg: dict, jobs: list[Job], output_root: Path,
             svd_paths: dict[str, Path]) -> list[tuple[Job, int]]:
    sched = cfg["scheduler"]
    per_gpu = int(sched.get("max_concurrent_per_gpu", 2))
    ds_sems = {
        name: threading.Semaphore(int(spec["max_concurrent"]))
        for name, spec in cfg["datasets"].items() if "max_concurrent" in spec
    }
    queues: dict[int, list[Job]] = defaultdict(list)
    for j in jobs:
        queues[j.gpu].append(j)

    results: list[tuple[Job, int]] = []
    lock = threading.Lock()

    def worker(gpu: int, queue: list[Job]):
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop(0)
            sem = ds_sems.get(job.dataset)
            if sem:
                sem.acquire()
            try:
                job_dir = job.dir(output_root)
                job_dir.mkdir(parents=True, exist_ok=True)
                cmd = build_command(cfg, job, output_root, svd_paths)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(job.gpu)}
                meta = {**job.key, "gpu": job.gpu, "cmd": cmd,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")}
                (job_dir / "job.json").write_text(json.dumps(meta, indent=2))
                print(f"[gpu{job.gpu}] START {job.variant}/{job.dataset}/"
                      f"{job.target}/seed{job.seed}")
                t0 = time.time()
                with open(job_dir / "train.log", "w") as f:
                    rc = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=f,
                                        stderr=subprocess.STDOUT).returncode
                meta.update(rc=rc, minutes=round((time.time() - t0) / 60, 1),
                            finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                (job_dir / "job.json").write_text(json.dumps(meta, indent=2))
                status = "DONE" if rc == 0 else f"FAILED rc={rc}"
                print(f"[gpu{job.gpu}] {status} {job.variant}/{job.dataset}/"
                      f"{job.target}/seed{job.seed} ({meta['minutes']} min)")
                with lock:
                    results.append((job, rc))
            finally:
                if sem:
                    sem.release()

    threads = []
    for gpu, queue in queues.items():
        for _ in range(per_gpu):
            t = threading.Thread(target=worker, args=(gpu, queue), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return results


def collate(cfg: dict, output_root: Path) -> None:
    rows = []
    test_ids: dict[tuple, dict[str, set]] = defaultdict(dict)
    ood_ids: dict[tuple, dict[str, set]] = defaultdict(dict)
    train_sizes = cfg.get("train_sizes") or [None]
    holdout_geos = cfg.get("holdout_geologies") or [None]
    for vname, vspec in cfg["variants"].items():
        for dname in cfg["datasets"]:
            for target in cfg["targets"]:
                for seed in cfg["seeds"]:
                    for size in train_sizes:
                        for hgeo in holdout_geos:
                            job = Job(vname, dname, target, int(seed),
                                      vspec["kind"], 0,
                                      train_size=size, holdout_geo=hgeo)
                            mp = job.metrics_path(output_root)
                            if not mp.exists():
                                continue
                            report = json.loads(mp.read_text())
                            for split, sr in report["splits"].items():
                                rows.append({
                                    "variant": vname, "dataset": dname,
                                    "target": target, "seed": seed,
                                    "train_size": size if size is not None else "full",
                                    "holdout_geo": hgeo if hgeo is not None else "",
                                    "split": split,
                                    "case_count": sr["case_count"],
                                    "sample_count": sr["sample_count"],
                                    **{k: round(float(v), 6)
                                       for k, v in sr["metrics"].items()},
                                })
                            tag = f"{vname}/{job.name}"
                            pred_csv = mp.parent / "test_predictions.csv"
                            if pred_csv.exists():
                                with open(pred_csv) as f:
                                    ids = {r["case_id"] for r in csv.DictReader(f)}
                                test_ids[(dname, target, hgeo)][tag] = ids
                            ood_csv = mp.parent / "test_ood_predictions.csv"
                            if ood_csv.exists():
                                with open(ood_csv) as f:
                                    ids = {r["case_id"] for r in csv.DictReader(f)}
                                ood_ids[(dname, target, hgeo)][tag] = ids

    if not rows:
        print("[collate] no completed runs found")
        return
    out_csv = output_root / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[collate] wrote {len(rows)} rows to {out_csv}")

    # split-identity assertion: same test case-id set across variants/seeds/
    # sizes, and across targets within a (dataset, holdout-fold) group
    # (geology-only stratification; subsampling never touches val/test).
    ok = True
    ref_by_group: dict[tuple, set] = {}
    for (dname, target, hgeo), variants in sorted(
            test_ids.items(), key=lambda kv: str(kv[0])):
        ref = next(iter(variants.values()))
        for name, ids in variants.items():
            if ids != ref:
                print(f"[collate] SPLIT MISMATCH {dname}/{target}/geo{hgeo}: "
                      f"{name} differs by {len(ids ^ ref)} case ids")
                ok = False
        gkey = (dname, hgeo)
        if gkey in ref_by_group and ref != ref_by_group[gkey]:
            print(f"[collate] SPLIT MISMATCH across targets in {dname}/geo{hgeo}")
            ok = False
        ref_by_group.setdefault(gkey, ref)
    # test_ood must be exactly the held-out geology's cases, identical across
    # variants, and disjoint from the in-dist test split.
    for (dname, target, hgeo), variants in sorted(
            ood_ids.items(), key=lambda kv: str(kv[0])):
        ref = next(iter(variants.values()))
        for name, ids in variants.items():
            if ids != ref:
                print(f"[collate] OOD MISMATCH {dname}/{target}/geo{hgeo}: {name}")
                ok = False
        indist = test_ids.get((dname, target, hgeo), {})
        if indist and (ref & next(iter(indist.values()))):
            print(f"[collate] OOD OVERLAPS in-dist test: {dname}/geo{hgeo}")
            ok = False
    print(f"[collate] split identity: {'PASS' if ok else 'FAIL'}")

    # pivots: one block per (train_size, holdout_geo, eval split) combination
    short_t = {"graph_energy_total": "energy",
               "graph_discounted_net_revenue": "revenue",
               "node_wept_final": "wept"}
    blocks = sorted({(r["train_size"], r["holdout_geo"]) for r in rows},
                    key=str)
    for size, hgeo in blocks:
        eval_splits = ["test", "test_ood"] if hgeo != "" else ["test"]
        for esplit in eval_splits:
            cells: dict[tuple, list] = defaultdict(list)
            for r in rows:
                if (r["split"] == esplit and r["train_size"] == size
                        and r["holdout_geo"] == hgeo):
                    cells[(r["variant"], r["dataset"], r["target"])].append(r)
            if not cells:
                continue
            cols = [(d, t) for d in cfg["datasets"] for t in cfg["targets"]]
            header = f"{'variant':<20}" + "".join(
                f"{d[:8]}/{short_t.get(t, t)[:7]:<9}"[:18].ljust(18)
                for d, t in cols)
            label = f"n={size}" + (f", holdout geo {hgeo}" if hgeo != "" else "")
            print(f"\n{esplit.upper()} MAPE% / R2 (mean over seeds) [{label}]")
            print(header)
            for vname in cfg["variants"]:
                line = f"{vname:<20}"
                for d, t in cols:
                    rs = cells.get((vname, d, t), [])
                    if rs:
                        mape = sum(x["mape"] for x in rs) / len(rs)
                        r2 = sum(x["r2"] for x in rs) / len(rs)
                        line += f"{mape:5.1f}/{r2:5.2f}     ".ljust(18)
                    else:
                        line += f"{'—':<18}"
                print(line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--collate-only", action="store_true")
    p.add_argument("--only", default=None,
                   help="comma-separated k=v filters, e.g. variant=mlp,dataset=simple_2pair")
    args = p.parse_args()

    cfg = load_config(args.config)
    output_root = (REPO / cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.collate_only:
        collate(cfg, output_root)
        return 0

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[matrix] WARNING: parent CUDA_VISIBLE_DEVICES="
              f"{os.environ['CUDA_VISIBLE_DEVICES']!r} is overridden per job; "
              f"config gpu pins refer to PHYSICAL device indices.")

    jobs = expand_jobs(cfg)
    if args.only:
        filt = dict(kv.split("=") for kv in args.only.split(","))
        jobs = [j for j in jobs if match(filt, j)]
    svd_paths = ensure_svd(cfg, jobs, output_root, args.dry_run)
    expected = {id(j): build_command(cfg, j, output_root, svd_paths) for j in jobs}
    pending = [j for j in jobs if not j.done(output_root, expected[id(j)])]

    print(f"[matrix] {len(jobs)} jobs, {len(jobs) - len(pending)} done, "
          f"{len(pending)} pending")
    if args.dry_run:
        for j in jobs:
            state = "done   " if j not in pending else "pending"
            print(f"  [{state}] gpu{j.gpu} {j.variant}/{j.dataset}/{j.target}"
                  f"/seed{j.seed}\n            {shlex.join(expected[id(j)])}")
        return 0

    failed = []
    if pending:
        results = run_jobs(cfg, pending, output_root, svd_paths)
        failed = [(j, rc) for j, rc in results if rc != 0]
        if failed:
            print(f"\n[matrix] {len(failed)} FAILED jobs:")
            for j, rc in failed:
                print(f"  rc={rc} {j.variant}/{j.dataset}/{j.target}/seed{j.seed} "
                      f"→ {j.dir(output_root) / 'train.log'}")
    collate(cfg, output_root)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
