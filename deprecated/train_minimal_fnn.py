from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FittedPreprocessors:
    feature_scaler: StandardScaler
    feature_pca: PCA | None
    target_scaler: StandardScaler


def load_dataset(h5_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    case_ids: list[str] = []
    n_padded = 0
    n_truncated = 0

    def encode_wells_fixed_12(wells: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        x = wells["x"].astype(np.float32)
        y = wells["y"].astype(np.float32)
        depth = wells["depth"].astype(np.float32)
        inj_rate = wells["inj_rate"].astype(np.float32)

        is_injector = (inj_rate > 0).astype(np.float32)
        per_well = np.stack([x, y, depth, is_injector], axis=1)

        if len(per_well) > 0:
            order = np.lexsort((depth, y, x))
            per_well = per_well[order]

        was_padded = False
        was_truncated = False
        if per_well.shape[0] < 12:
            pad_rows = 12 - per_well.shape[0]
            per_well = np.vstack([per_well, np.zeros((pad_rows, 4), dtype=np.float32)])
            was_padded = True
        elif per_well.shape[0] > 12:
            per_well = per_well[:12]
            was_truncated = True

        return per_well.reshape(-1), was_padded, was_truncated

    with h5py.File(h5_path, "r") as handle:
        for case_id in sorted(handle.keys()):
            group = handle[case_id]
            wells = group["wells"][:]
            production_total = group["outputs"]["field_energy_production_total"][:]

            flat, was_padded, was_truncated = encode_wells_fixed_12(wells)
            if was_padded:
                n_padded += 1
            if was_truncated:
                n_truncated += 1

            features.append(flat)
            targets.append(float(production_total[-1]))
            case_ids.append(case_id)

    if n_padded > 0 or n_truncated > 0:
        print(
            "Adjusted non-12-well cases: "
            f"padded={n_padded}, truncated={n_truncated}"
        )

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32), case_ids


def split_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0 or test_fraction <= 0:
        raise ValueError("val_fraction and test_fraction must be > 0.")
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("val_fraction + test_fraction is too large for stable training.")

    all_indices = np.arange(n_samples)
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
    )

    val_relative = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def fit_preprocessors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    whiten: bool,
    n_components: int | None,
) -> FittedPreprocessors:
    feature_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train)

    feature_pca: PCA | None = None
    if whiten:
        feature_pca = PCA(n_components=n_components, whiten=True, random_state=42)
        feature_pca.fit(x_train_scaled)

    target_scaler = StandardScaler()
    target_scaler.fit(y_train.reshape(-1, 1))

    return FittedPreprocessors(
        feature_scaler=feature_scaler,
        feature_pca=feature_pca,
        target_scaler=target_scaler,
    )


def transform_features(x: np.ndarray, preprocessors: FittedPreprocessors) -> np.ndarray:
    x_scaled = preprocessors.feature_scaler.transform(x)
    if preprocessors.feature_pca is not None:
        x_scaled = preprocessors.feature_pca.transform(x_scaled)
    return x_scaled.astype(np.float32)


def transform_targets(y: np.ndarray, preprocessors: FittedPreprocessors) -> np.ndarray:
    y_scaled = preprocessors.target_scaler.transform(y.reshape(-1, 1)).reshape(-1)
    return y_scaled.astype(np.float32)


class GeothermalDataModule(L.LightningDataModule):
    def __init__(
        self,
        x: np.ndarray,
        y_scaled: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.x = x
        self.y_scaled = y_scaled
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = TensorDataset(
            torch.from_numpy(self.x[self.train_idx]),
            torch.from_numpy(self.y_scaled[self.train_idx]),
        )
        self.val_ds = TensorDataset(
            torch.from_numpy(self.x[self.val_idx]),
            torch.from_numpy(self.y_scaled[self.val_idx]),
        )
        self.test_ds = TensorDataset(
            torch.from_numpy(self.x[self.test_idx]),
            torch.from_numpy(self.y_scaled[self.test_idx]),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )


class FNNRegressor(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_cls = activations.get(activation.lower())
        if activation_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    activation_cls(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        mae = torch.mean(torch.abs(preds - y))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_mae_scaled", mae, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

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
            patience=12,
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


def predict_scaled(model: FNNRegressor, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).to(device)
        pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
    return pred.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_error_scatter_plots(
    output_dir: Path,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residual = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))

    axes[0].scatter(y_true, y_pred, s=18, alpha=0.7)
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axes[0].set_xlabel("Actual final field_energy_production_total")
    axes[0].set_ylabel("Predicted final field_energy_production_total")
    axes[0].set_title(f"{split_name}: Predicted vs Actual")

    axes[1].scatter(y_true, residual, s=18, alpha=0.7)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_xlabel("Actual final field_energy_production_total")
    axes[1].set_ylabel("Prediction error (pred - actual)")
    axes[1].set_title(f"{split_name}: Error Scatter")

    fig.tight_layout()
    figure_path = output_dir / f"{split_name}_prediction_error_scatter.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Saved plot: {figure_path}")


def save_loss_curve_plot(metrics_csv_path: Path, output_path: Path) -> None:
    if not metrics_csv_path.exists():
        print(f"Skipping loss plot; metrics file not found: {metrics_csv_path}")
        return

    train_epochs: list[float] = []
    train_losses: list[float] = []
    val_epochs: list[float] = []
    val_losses: list[float] = []

    with metrics_csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch_raw = row.get("epoch", "")
            if epoch_raw is None or epoch_raw == "":
                continue

            try:
                epoch = float(epoch_raw)
            except ValueError:
                continue

            train_raw = row.get("train_loss_epoch", "")
            if train_raw not in (None, ""):
                try:
                    train_epochs.append(epoch)
                    train_losses.append(float(train_raw))
                except ValueError:
                    pass

            val_raw = row.get("val_loss", "")
            if val_raw not in (None, ""):
                try:
                    val_epochs.append(epoch)
                    val_losses.append(float(val_raw))
                except ValueError:
                    pass

    if not train_losses and not val_losses:
        print("Skipping loss plot; no usable loss rows found in metrics.csv")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_losses:
        ax.plot(train_epochs, train_losses, label="train_loss_epoch")
    if val_losses:
        ax.plot(val_epochs, val_losses, label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, scaled target)")
    ax.set_title("Loss Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def parse_hidden_dims(hidden_dims_raw: str) -> tuple[int, ...]:
    values = [int(v.strip()) for v in hidden_dims_raw.split(",") if v.strip()]
    if not values:
        raise ValueError("hidden_dims must include at least one positive integer.")
    if any(v <= 0 for v in values):
        raise ValueError("hidden_dims values must be positive.")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch Lightning FNN to predict final field_energy_production_total."
    )
    parser.add_argument("--h5-path", type=Path, default=Path("minimal_compiled_all.h5"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--activation", type=str, choices=["relu", "gelu", "silu"], default="gelu")
    parser.add_argument("--hidden-dims", type=str, default="256,128,64")
    parser.add_argument("--no-whiten", action="store_true", help="Disable PCA whitening on features.")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Optional number of PCA components when whitening is enabled.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Optional output directory for scatter plots. Defaults to <logger_run_dir>/plots.",
    )
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    x_raw, y_raw, case_ids = load_dataset(args.h5_path)
    train_idx, val_idx, test_idx = split_indices(
        n_samples=len(case_ids),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    preprocessors = fit_preprocessors(
        x_train=x_raw[train_idx],
        y_train=y_raw[train_idx],
        whiten=not args.no_whiten,
        n_components=args.pca_components,
    )

    x_processed = transform_features(x_raw, preprocessors)
    y_scaled = transform_targets(y_raw, preprocessors)

    input_dim = x_processed.shape[1]
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    print(f"Loaded {len(case_ids)} cases from {args.h5_path}")
    print(f"Input features after scaling/whitening: {input_dim}")
    print(
        "Split sizes: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    data_module = GeothermalDataModule(
        x=x_processed,
        y_scaled=y_scaled,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = FNNRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=args.activation,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:03d}-{val_loss:.4f}",
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.early_stop_patience,
        min_delta=1e-5,
    )
    logger = CSVLogger(save_dir="lightning_logs", name="geothermal_fnn")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model=model, datamodule=data_module)

    best_model_path = checkpoint_cb.best_model_path
    if not best_model_path:
        raise RuntimeError("No checkpoint was saved. Training likely failed.")

    print(f"Best checkpoint: {best_model_path}")

    best_model = FNNRegressor.load_from_checkpoint(best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("mps")
    best_model = best_model.to(device)

    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    print("\nMetrics in original target units:")
    if args.plots_dir is None:
        plots_dir = Path(logger.log_dir) / "plots"
    else:
        plots_dir = args.plots_dir

    for split_name, split_idx in split_map.items():
        preds_scaled = predict_scaled(best_model, x_processed[split_idx], device)
        preds = preprocessors.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        actual = y_raw[split_idx]
        metrics = compute_metrics(actual, preds)
        print(
            f"  {split_name:<5} | MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
        )
        save_error_scatter_plots(plots_dir, split_name, actual, preds)

    metrics_csv_path = Path(logger.log_dir) / "metrics.csv"
    loss_plot_path = plots_dir / "loss_over_time.png"
    save_loss_curve_plot(metrics_csv_path, loss_plot_path)


if __name__ == "__main__":
    main()
