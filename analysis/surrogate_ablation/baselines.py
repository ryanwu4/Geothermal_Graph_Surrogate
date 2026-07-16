"""Baseline models for the surrogate-architecture ablation study.

Two non-GNN reference architectures, trained by train_baseline.py through the
SAME data pipeline as train.py (same load_hetero_graphs call, same
geology-stratified split, same HeteroGraphScaler target whitening) so results
are directly comparable to the GNN variants:

- WellMLPBaseline: flattened per-well [x, y, depth, is_inj] (canonical well
  order) + 15-d geology one-hot -> MLP. The floor baseline: no geology grid,
  no relational structure.
- GlobalCNN3D: full-grid 3D CNN over the 7 physics channels + 2 well-marker
  channels (Gaussian cylinder heatmaps, injectors/producers separately).
  Graph-level head via dual adaptive pooling; node-level head via grid_sample
  readout of the last feature map at well positions.

Axis frame (established for these datasets): physics volume is (Z=63, X=70,
Y=76); well 'x' indexes the 70-axis, 'y' the 76-axis, 'depth' the Z axis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geothermal.data import (  # noqa: E402
    HeteroGraphScaler,
    load_hetero_graphs,
    split_indices_stratified,
    subsample_train_indices,
)

N_GEOLOGIES = 15
PHYSICS_CHANNELS = [
    "PermX", "PermY", "PermZ", "Porosity", "Temperature0", "Pressure0", "valid_mask",
]


# --------------- shared data prep (split parity with train.py) ---------------


def load_split_scale(
    h5_path: Path,
    target: str,
    split_seed: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_cases: int | None = None,
    holdout_geologies: list[int] | None = None,
    train_subsample: int | None = None,
    subsample_seed: int | None = None,
):
    """Replicates train.py's load + geology-stratified split + scaler fit.

    Returns (graphs_raw, split_indices dict, scaler, geology_indices).
    Requires a complete case_geology_map.json adjacent to the H5 (all three
    ablation datasets have one) — errors out rather than silently falling back
    to a different stratification than the GNN runs.

    holdout_geologies: cases of these geologies are excluded from train/val/
    test and returned as an extra 'test_ood' split (mirrors train.py's
    --holdout-geologies). train_subsample: geology-proportional subsample of
    the train split, seeded by subsample_seed (mirrors --train-subsample);
    scaler is fit on the subsample.
    """
    graphs_raw, targets = load_hetero_graphs(
        h5_path, target=target, node_encoder="cnn", enrich_global_attr=True,
        max_cases=max_cases,
    )
    case_ids = [g.case_id for g in graphs_raw]

    gmap_path = Path(h5_path).parent / "case_geology_map.json"
    with open(gmap_path) as f:
        gmap = json.load(f)
    missing = [cid for cid in case_ids if cid not in gmap]
    if missing:
        raise RuntimeError(
            f"{gmap_path} missing {len(missing)} case ids (e.g. {missing[0]}); "
            "baselines require the same geology-stratified split as train.py."
        )
    geology_indices = np.array(
        [int(gmap[cid]["geology_index"]) for cid in case_ids], dtype=np.int64
    )

    splits: dict[str, np.ndarray] = {}
    if holdout_geologies:
        all_idx = np.arange(len(graphs_raw))
        ood_mask = np.isin(geology_indices, sorted(set(holdout_geologies)))
        if not ood_mask.any():
            raise RuntimeError(f"holdout {holdout_geologies}: no cases matched")
        indist = all_idx[~ood_mask]
        tr_l, va_l, te_l = split_indices_stratified(
            targets=targets[indist],
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
            geology_indices=geology_indices[indist],
        )
        splits = {"train": indist[np.asarray(tr_l)],
                  "val": indist[np.asarray(va_l)],
                  "test": indist[np.asarray(te_l)],
                  "test_ood": all_idx[ood_mask]}
    else:
        train_idx, val_idx, test_idx = split_indices_stratified(
            targets=targets,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
            geology_indices=geology_indices,
        )
        splits = {"train": np.asarray(train_idx), "val": np.asarray(val_idx),
                  "test": np.asarray(test_idx)}

    if train_subsample is not None:
        assert subsample_seed is not None, "train_subsample needs subsample_seed"
        n_before = len(splits["train"])
        splits["train"] = subsample_train_indices(
            splits["train"], geology_indices, train_subsample, seed=subsample_seed
        )
        print(f"[baseline] train subsample: {n_before} -> {len(splits['train'])} "
              f"(seed {subsample_seed})")

    scaler = HeteroGraphScaler(whiten=True, pca_components=None)
    scaler.fit([graphs_raw[i] for i in splits["train"]])
    return graphs_raw, splits, scaler, geology_indices


def canonical_well_order(pos_xyz: torch.Tensor, is_injector: torch.Tensor) -> np.ndarray:
    """Injectors first, then producers, each sorted by (x, y). Returns index array."""
    pos = pos_xyz.cpu().numpy()
    inj = is_injector.cpu().numpy() > 0.5
    keys = [(-int(inj[i]), pos[i, 0], pos[i, 1], pos[i, 2]) for i in range(len(inj))]
    return np.array(sorted(range(len(inj)), key=lambda i: keys[i]), dtype=np.int64)


def build_case_arrays(graph, geology_index: int, target_scaler):
    """Per-case arrays in canonical well order.

    Returns dict with: feats (n_wells, 4) raw [x,y,depth,is_inj], geo_onehot
    (15,), y_scaled (out_dim,) or (n_wells, out) for node targets, prod_mask
    (n_wells,) bool, case_id.
    """
    order = canonical_well_order(graph["well"].pos_xyz, graph["well"].is_injector)
    pos = graph["well"].pos_xyz.cpu().numpy()[order]
    inj = (graph["well"].is_injector.cpu().numpy()[order] > 0.5).astype(np.float32)
    feats = np.concatenate([pos, inj[:, None]], axis=1).astype(np.float32)
    onehot = np.zeros(N_GEOLOGIES, dtype=np.float32)
    onehot[geology_index] = 1.0

    y_raw = graph.y.cpu().numpy()
    node_level = getattr(graph, "prediction_level", "graph") == "node"
    y_scaled = target_scaler.transform(y_raw).astype(np.float32)
    if node_level:
        y_scaled = y_scaled[order]
    return {
        "feats": feats,
        "geo_onehot": onehot,
        "y_scaled": y_scaled,
        "prod_mask": inj < 0.5,
        "node_level": node_level,
        "case_id": graph.case_id,
    }


# --------------- MLP baseline ---------------


class WellMLPBaseline(L.LightningModule):
    """Flattened well features + geology one-hot -> MLP.

    Graph targets: 1 output. Node targets: n_wells outputs in canonical well
    order, loss/eval masked to producer slots (mirrors filter_extractors).
    """

    def __init__(
        self,
        n_wells: int,
        prediction_level: str = "graph",
        hidden_dims: tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        input_dim = n_wells * 4 + N_GEOLOGIES
        output_dim = n_wells if prediction_level == "node" else 1
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage: str):
        x, y, mask = batch
        pred = self(x)
        if self.hparams.prediction_level == "node":
            loss = self.loss_fn(pred[mask], y[mask])
        else:
            loss = self.loss_fn(pred, y)
        self.log(f"{stage}_loss", loss, prog_bar=stage == "val", batch_size=x.shape[0])
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


# --------------- Global 3D CNN baseline ---------------


def well_marker_channels(
    pos: torch.Tensor, is_inj: torch.Tensor, vol_shape: tuple[int, int, int],
    sigma: float = 2.0,
) -> torch.Tensor:
    """(B, 2, Z, X, Y) Gaussian cylinder heatmaps: ch0=injectors, ch1=producers.

    Cylinder semantics follow physics_slab.generate_3d_heatmaps: full strength
    from the surface down to the well depth, Gaussian decay below; Gaussian
    radial falloff in (X, Y). pos is (B, n, 3) = (x in X-axis, y in Y-axis,
    depth in Z-axis) in GLOBAL grid coords.
    """
    B, n, _ = pos.shape
    Z, X, Y = vol_shape
    dev = pos.device
    xs = torch.arange(X, device=dev, dtype=torch.float32)
    ys = torch.arange(Y, device=dev, dtype=torch.float32)
    zs = torch.arange(Z, device=dev, dtype=torch.float32)

    px = pos[..., 0].unsqueeze(-1)                       # (B, n, 1)
    py = pos[..., 1].unsqueeze(-1)
    pz = pos[..., 2].unsqueeze(-1)
    two_sig2 = 2.0 * sigma * sigma
    gx = torch.exp(-((xs - px) ** 2) / two_sig2)         # (B, n, X)
    gy = torch.exp(-((ys - py) ** 2) / two_sig2)         # (B, n, Y)
    below = zs - pz                                      # (B, n, Z)
    gz = torch.where(
        below <= 0, torch.ones_like(below), torch.exp(-(below ** 2) / two_sig2)
    )                                                    # (B, n, Z)

    heat = (
        gz[:, :, :, None, None]
        * gx[:, :, None, :, None]
        * gy[:, :, None, None, :]
    )                                                    # (B, n, Z, X, Y)
    inj_w = (is_inj > 0.5).float()[:, :, None, None, None]
    ch_inj = (heat * inj_w).sum(dim=1).clamp(max=1.0)
    ch_prod = (heat * (1.0 - inj_w)).sum(dim=1).clamp(max=1.0)
    return torch.stack([ch_inj, ch_prod], dim=1)         # (B, 2, Z, X, Y)


class GlobalCNN3D(L.LightningModule):
    """Lightweight 3D CNN over the full geology grid + well-marker channels.

    Node-level readout: grid_sample of the final feature map at well positions
    (grid coord order (W, H, D) = (y, x, z), matching ContinuousCropper's
    convention), concatenated with raw per-well features, through a shared MLP.
    Fallback if this underfits: crop a local patch of the feature map per well
    instead of a point read.
    """

    def __init__(
        self,
        prediction_level: str = "graph",
        widths: tuple[int, ...] = (16, 32, 64, 128),
        marker_sigma: float = 2.0,
        vol_shape: tuple[int, int, int] = (63, 70, 76),
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()
        in_ch = len(PHYSICS_CHANNELS) + 2
        convs: list[nn.Module] = []
        prev = in_ch
        for w in widths:
            convs += [
                nn.Conv3d(prev, w, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, w),
                nn.GELU(),
            ]
            prev = w
        self.body = nn.Sequential(*convs)
        feat = widths[-1]
        self.graph_head = nn.Sequential(
            nn.Linear(2 * feat, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.node_head = nn.Sequential(
            nn.Linear(feat + 4, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, vol7, pos, is_inj):
        markers = well_marker_channels(
            pos, is_inj, tuple(self.hparams.vol_shape), self.hparams.marker_sigma
        )
        x = torch.cat([vol7, markers], dim=1)            # (B, 9, Z, X, Y)
        fmap = self.body(x)                              # (B, C, z', x', y')
        if self.hparams.prediction_level == "graph":
            avg = F.adaptive_avg_pool3d(fmap, 1).flatten(1)
            mx = F.adaptive_max_pool3d(fmap, 1).flatten(1)
            return self.graph_head(torch.cat([avg, mx], dim=1))   # (B, 1)
        # node-level: grid_sample point readout at well positions.
        Z, X, Y = self.hparams.vol_shape
        norm = lambda v, n: 2.0 * v / (n - 1) - 1.0
        grid = torch.stack(
            [norm(pos[..., 1], Y), norm(pos[..., 0], X), norm(pos[..., 2], Z)],
            dim=-1,
        )                                                # (B, n, 3) as (w, h, d)
        grid = grid[:, :, None, None, :]                 # (B, n, 1, 1, 3)
        sampled = F.grid_sample(fmap, grid, align_corners=True)   # (B, C, n, 1, 1)
        sampled = sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (B, n, C)
        raw = torch.cat([pos, is_inj.unsqueeze(-1)], dim=-1)        # (B, n, 4)
        return self.node_head(torch.cat([sampled, raw], dim=-1)).squeeze(-1)  # (B, n)

    def _step(self, batch, stage: str):
        vol7, pos, is_inj, y, mask = batch
        pred = self(vol7, pos, is_inj)
        if self.hparams.prediction_level == "node":
            loss = self.loss_fn(pred[mask], y[mask])
        else:
            loss = self.loss_fn(pred, y)
        self.log(f"{stage}_loss", loss, prog_bar=stage == "val",
                 batch_size=vol7.shape[0])
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }


# --------------- torch Datasets ---------------


class MLPDataset(torch.utils.data.Dataset):
    """Precomputed flat tensors. X standardized with train-split stats."""

    def __init__(self, X, Y, M, case_ids):
        self.X, self.Y, self.M = X, Y, M
        self.case_ids = case_ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.M[i]


class VolumeDataset(torch.utils.data.Dataset):
    """Per-case physics volume + well tensors for GlobalCNN3D."""

    def __init__(self, graphs, cases, device=None):
        self.vols, self.pos, self.inj, self.ys, self.masks, self.case_ids = \
            [], [], [], [], [], []
        for g, c in zip(graphs, cases):
            vol = torch.stack([g.physics_context.d[k] for k in PHYSICS_CHANNELS])
            assert tuple(vol.shape[1:]) == (63, 70, 76), (
                f"{c['case_id']}: volume {tuple(vol.shape[1:])} != GlobalCNN3D's "
                "vol_shape (63,70,76); well markers/readout would be misplaced"
            )
            order = canonical_well_order(g["well"].pos_xyz, g["well"].is_injector)
            pos = g["well"].pos_xyz[order]
            inj = g["well"].is_injector[order]
            y = torch.from_numpy(c["y_scaled"])
            y = y.squeeze(-1) if c["node_level"] else y.reshape(1)
            mask = torch.from_numpy(c["prod_mask"])
            if device is not None:
                vol, pos, inj, y, mask = (t.to(device) for t in (vol, pos, inj, y, mask))
            self.vols.append(vol)
            self.pos.append(pos)
            self.inj.append(inj)
            self.ys.append(y.float())
            self.masks.append(mask)
            self.case_ids.append(c["case_id"])

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, i):
        return self.vols[i], self.pos[i], self.inj[i], self.ys[i], self.masks[i]


def grid_sample_unit_check(device: str = "cpu") -> None:
    """Guard the (w,h,d)=(y,x,z) grid_sample convention with a delta volume."""
    Z, X, Y = 16, 20, 24
    well = torch.tensor([[[5.0, 7.0, 9.0]]], device=device)   # x=5, y=7, z=9
    vol = torch.zeros(1, 1, Z, X, Y, device=device)
    vol[0, 0, 9, 5, 7] = 1.0
    norm = lambda v, n: 2.0 * v / (n - 1) - 1.0
    grid = torch.stack(
        [norm(well[..., 1], Y), norm(well[..., 0], X), norm(well[..., 2], Z)], dim=-1
    )[:, :, None, None, :]
    out = F.grid_sample(vol, grid, align_corners=True)
    val = float(out.flatten()[0])
    assert abs(val - 1.0) < 1e-5, f"grid_sample axis convention broken: {val}"


if __name__ == "__main__":
    grid_sample_unit_check()
    print("grid_sample unit check PASS")
