"""HeteroGNN model definition, constants, and seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import torch
import lightning as L
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    BatchNorm,
    NNConv,
    HeteroConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    LayerNorm,
)

# --------------- constants ---------------

EDGE_TYPES = [
    ("well", "inj_to_inj", "well"),
    ("well", "ext_to_ext", "well"),
    ("well", "inj_to_ext", "well"),
    ("well", "ext_to_inj", "well"),
]

# Number of T/P profile statistics: mean, min, max for T and P
TP_PROFILE_STATS = 6


# --------------- seeding ---------------


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------- model ---------------


class HeteroGNNRegressor(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        global_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pooling: str,
        residual: bool,
        learning_rate: float,
        weight_decay: float,
        loss: str,
        prediction_level: str = "graph",  # "graph" or "node"
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.loss_type = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.prediction_level = prediction_level

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_proj = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in EDGE_TYPES:
                e_dim = 14
                nn_edge = nn.Sequential(
                    nn.Linear(e_dim, 32),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, in_dim * hidden_dim),
                )
                conv_dict[edge_type] = NNConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    nn=nn_edge,
                    aggr="mean",
                    root_weight=True,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(LayerNorm(hidden_dim))
            if in_dim == hidden_dim:
                self.residual_proj.append(nn.Identity())
            else:
                self.residual_proj.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)

        if prediction_level == "graph":
            pool_dim = 3 * hidden_dim
            head_input_dim = pool_dim + global_dim
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, self.output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, self.output_dim),
            )

        if loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss == "huber":
            self.loss_fn = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = {"well": batch["well"].x}
        edge_index_dict = {
            edge_type: batch[edge_type].edge_index for edge_type in EDGE_TYPES
        }
        edge_attr_dict = {
            edge_type: batch[edge_type].edge_attr for edge_type in EDGE_TYPES
        }

        x = x_dict["well"]

        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x
            out_dict = conv(
                x_dict={"well": x},
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
            x = out_dict["well"]
            x = norm(x, batch["well"].batch)
            x = self.activation(x)
            x = self.dropout_layer(x)
            if self.hparams.residual:
                x = x + self.residual_proj[idx](x_in)

        if self.prediction_level == "graph":
            node_batch = batch["well"].batch
            x_sum = global_add_pool(x, node_batch)
            x_mean = global_mean_pool(x, node_batch)
            x_max = global_max_pool(x, node_batch)
            graph_embed = torch.cat([x_sum, x_mean, x_max], dim=-1)
            global_feat = batch.global_attr
            graph_embed = torch.cat([graph_embed, global_feat], dim=-1)
            return self.head(graph_embed)  # (B, 1)
        else:
            return self.head(x)  # (N_nodes, 1)

    def _step(self, batch: HeteroData, stage: str) -> torch.Tensor:
        preds = self(batch)
        target = batch.y

        filter_ext = getattr(batch, "filter_extractors", True)
        if isinstance(filter_ext, torch.Tensor):
            should_filter = bool(filter_ext[0].item())
        elif isinstance(filter_ext, list):
            should_filter = bool(filter_ext[0])
        else:
            should_filter = bool(filter_ext)

        if self.prediction_level == "node" and should_filter:
            ext_mask = batch["well"].is_injector < 0.5
            preds = preds[ext_mask]
            target = target[ext_mask]

        loss = self.loss_fn(preds, target)
        mae = torch.mean(torch.abs(preds - target))
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self.log(
            f"{stage}_mae_scaled",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch.num_graphs,
        )
        return loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
