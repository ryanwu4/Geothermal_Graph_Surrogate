import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousCropper(nn.Module):
    """
    Dynamically extracts a continuous 3D bounding box between two arbitrary well coordinates.
    Uses affine_grid and grid_sample for fully differentiable interpolation.
    """
    def __init__(self, out_shape: tuple[int, int, int] = (16, 32, 32)):
        """
        Args:
            out_shape: (D_out, H_out, W_out) corresponding to (Z, X, Y).
                       Default is 16 depth, 32 X, 32 Y.
        """
        super().__init__()
        self.out_shape = out_shape  # (Z, X, Y)

    def forward(self, 
                volumes: torch.Tensor, 
                coords_a: torch.Tensor, 
                coords_b: torch.Tensor, 
                full_shape: tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            volumes: (B, C, Z, X, Y)
            coords_a: (B, 3) continuous coordinates of Well A in grid index space (X, Y, Z)
            coords_b: (B, 3) continuous coordinates of Well B in grid index space (X, Y, Z)
            full_shape: (Z_len, X_len, Y_len) the dimensions of the original grid index space
        Returns:
            sampled_slabs: (B, C, Z_out, X_out, Y_out)
            bounds: bounds dict containing local well heatmaps info
        """
        B, C, Z_dim, X_dim, Y_dim = volumes.shape
        Z_len, X_len, Y_len = full_shape

        x_a, y_a, z_a = coords_a[:, 0], coords_a[:, 1], coords_a[:, 2]
        x_b, y_b, z_b = coords_b[:, 0], coords_b[:, 1], coords_b[:, 2]

        # Calculate bounding box bounds (in continuous original space)
        x_min = torch.min(x_a, x_b) - 10.0
        x_max = torch.max(x_a, x_b) + 10.0
        y_min = torch.min(y_a, y_b) - 10.0
        y_max = torch.max(y_a, y_b) + 10.0
        
        # Z-dimension bounding (no +10 padding, Z surface is ALWAYS 0)
        # Bounding box includes from surface down to the deepest well coordinate + 1
        z_min = torch.zeros_like(z_a)
        z_max = torch.max(z_a, z_b)
        # Ensure minimum delta to avoid singular affine matrices
        z_max = torch.where(z_max - z_min < 1.0, z_min + 1.0, z_max)

        # Map to [-1, 1] relative to the actual tensor dimension we have right now (which might be cropped)
        # The input coordinates were against full_shape. 
        # So X in [0, X_len-1] maps to [-1, 1].
        x_min_norm = (x_min / (X_len - 1)) * 2 - 1
        x_max_norm = (x_max / (X_len - 1)) * 2 - 1
        y_min_norm = (y_min / (Y_len - 1)) * 2 - 1
        y_max_norm = (y_max / (Y_len - 1)) * 2 - 1
        z_min_norm = (z_min / (Z_len - 1)) * 2 - 1
        z_max_norm = (z_max / (Z_len - 1)) * 2 - 1

        # Affine matrix maps [-1,1] in output to [min_norm, max_norm] in input
        # theta is [B, 3, 4]
        # Mapping:
        # u = (max_norm - min_norm)/2 * u' + (max_norm + min_norm)/2
        theta = torch.zeros((B, 3, 4), dtype=volumes.dtype, device=volumes.device)
        
        # grid_sample treats dimensions as (W, H, D) -> (Y, X, Z) for 5D inputs
        # So:
        # row 0 maps to w (Y dimension)
        theta[:, 0, 0] = (y_max_norm - y_min_norm) / 2
        theta[:, 0, 3] = (y_max_norm + y_min_norm) / 2
        
        # row 1 maps to v (X dimension)
        theta[:, 1, 1] = (x_max_norm - x_min_norm) / 2
        theta[:, 1, 3] = (x_max_norm + x_min_norm) / 2
        
        # row 2 maps to u (Z dimension)
        theta[:, 2, 2] = (z_max_norm - z_min_norm) / 2
        theta[:, 2, 3] = (z_max_norm + z_min_norm) / 2

        grid = F.affine_grid(theta, size=(B, C, *self.out_shape), align_corners=True)
        sampled_slabs = F.grid_sample(volumes, grid, padding_mode='zeros', align_corners=True)

        return sampled_slabs, {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }


def generate_3d_heatmaps(coords_a: torch.Tensor, 
                         coords_b: torch.Tensor, 
                         bounds: dict,
                         out_shape: tuple[int, int, int],
                         sigma: float = 2.0) -> torch.Tensor:
    """
    Generates two 3D Gaussian heatmaps peaking at 1.0 at Well A and Well B.
    
    Args:
        coords_a/b: (B, 3) in full original space (X, Y, Z)
        bounds: dict from ContinuousCropper containing physical boundaries
        out_shape: (Z, X, Y)
    Returns:
        heatmaps: (B, 2, Z_out, X_out, Y_out)
    """
    B = coords_a.shape[0]
    Z_out, X_out, Y_out = out_shape
    device = coords_a.device
    dtype = coords_a.dtype

    # Map global coords to local [0, D-1] space
    x_len = bounds['x_max'] - bounds['x_min']
    y_len = bounds['y_max'] - bounds['y_min']
    z_len = bounds['z_max'] - bounds['z_min']

    local_x_a = (coords_a[:, 0] - bounds['x_min']) / x_len * (X_out - 1)
    local_y_a = (coords_a[:, 1] - bounds['y_min']) / y_len * (Y_out - 1)
    local_z_a = (coords_a[:, 2] - bounds['z_min']) / z_len * (Z_out - 1)

    local_x_b = (coords_b[:, 0] - bounds['x_min']) / x_len * (X_out - 1)
    local_y_b = (coords_b[:, 1] - bounds['y_min']) / y_len * (Y_out - 1)
    local_z_b = (coords_b[:, 2] - bounds['z_min']) / z_len * (Z_out - 1)

    # 3D Meshgrid
    z_grid = torch.arange(Z_out, dtype=dtype, device=device).view(1, Z_out, 1, 1).expand(B, Z_out, X_out, Y_out)
    x_grid = torch.arange(X_out, dtype=dtype, device=device).view(1, 1, X_out, 1).expand(B, Z_out, X_out, Y_out)
    y_grid = torch.arange(Y_out, dtype=dtype, device=device).view(1, 1, 1, Y_out).expand(B, Z_out, X_out, Y_out)

    # Well A heatmap (Vertical Cylinder)
    # 1. Horizontal distance squared
    dist_sq_xy_a = (x_grid - local_x_a.view(B, 1, 1, 1))**2 + (y_grid - local_y_a.view(B, 1, 1, 1))**2
    # 2. Vertical activation: Active from surface (0) down to well depth (local_z_a)
    # Below depth, it degrades according to a Gaussian distance
    dist_sq_z_a = torch.where(
        z_grid <= local_z_a.view(B, 1, 1, 1),
        torch.zeros_like(z_grid),
        (z_grid - local_z_a.view(B, 1, 1, 1))**2
    )
    dist_sq_a = dist_sq_xy_a + dist_sq_z_a
    heatmap_a = torch.exp(-dist_sq_a / (2 * sigma**2))

    # Well B heatmap (Vertical Cylinder)
    dist_sq_xy_b = (x_grid - local_x_b.view(B, 1, 1, 1))**2 + (y_grid - local_y_b.view(B, 1, 1, 1))**2
    dist_sq_z_b = torch.where(
        z_grid <= local_z_b.view(B, 1, 1, 1),
        torch.zeros_like(z_grid),
        (z_grid - local_z_b.view(B, 1, 1, 1))**2
    )
    dist_sq_b = dist_sq_xy_b + dist_sq_z_b
    heatmap_b = torch.exp(-dist_sq_b / (2 * sigma**2))

    return torch.stack([heatmap_a, heatmap_b], dim=1)


class PhysicsSlabExtractor(nn.Module):
    """
    Combines ContinuousCropper and heatmap generation. 
    Allows dynamic channel configuration.
    """
    def __init__(self, 
                 active_channels: list[str],
                 out_shape: tuple[int, int, int] = (16, 32, 32),
                 sigma: float = 2.0):
        super().__init__()
        self.cropper = ContinuousCropper(out_shape=out_shape)
        self.active_channels = active_channels
        self.out_shape = out_shape
        self.sigma = sigma
        
        # Valid channel mapping expected from the dataloader's multi-channel volume
        self.all_possible_channels = [
            "PermX", "PermY", "PermZ", "Porosity", "Temperature0", "Pressure0", "valid_mask"
        ]
        
    def forward(self, 
                volumes_dict: dict[str, torch.Tensor], 
                coords_a: torch.Tensor, 
                coords_b: torch.Tensor, 
                full_shape: tuple[int, int, int]) -> torch.Tensor:
        """
        Dynamically extracts and concatenates chosen channels.
        Volumes dict contains (B, Z, X, Y) or (B, 1, Z, X, Y) tensors.
        Returns:
            slab: (B, C, Z_out, X_out, Y_out)
        """
        # Form the multi-channel volume based on active_channels
        tensors = []
        device = coords_a.device
        for ch in self.active_channels:
            if ch in volumes_dict:
                t = volumes_dict[ch].to(device, non_blocking=True)
                if t.ndim == 4: # (B, Z, X, Y) -> (B, 1, Z, X, Y)
                    t = t.unsqueeze(1)
                tensors.append(t)
        
        multi_vol = torch.cat(tensors, dim=1) # (B, C_subset, Z, X, Y)
        
        # Grid sample the continuous physics crop
        slab, bounds = self.cropper(multi_vol, coords_a, coords_b, full_shape)

        # Generate Gaussian heatmaps 
        heatmaps = generate_3d_heatmaps(coords_a, coords_b, bounds, self.out_shape, self.sigma)
        
        # Concatenate heatmaps
        slab = torch.cat([slab, heatmaps], dim=1)
        
        return slab


class PhysicsSlabCNN(nn.Module):
    """
    Modular 3D CNN Edge Feature Extractor.
    Takes [B, C, Z=16, X=32, Y=32] volumetric slabs, extracts feature vectors,
    injects delta_s (continuous Euclidean distance), and outputs a configured latent edge dimension.

    Two opt-in changes (default off) mirror the v3 node-CNN fix that resolved geo 8 underfitting:
      - `norm="groupnorm"` replaces BatchNorm3d with GroupNorm. BN normalizes across
        the batch and strips inter-instance absolute-scale signal (e.g., "this geology's
        PermZ is tiny") — fine for in-distribution learning but destroys the OOD-defining
        signal for outlier geologies. GroupNorm normalizes per-instance only.
      - `raw_means_bypass=True` concatenates per-channel slab means (computed pre-CNN,
        never normalized) into the final MLP. Provides direct access to absolute-magnitude
        signal regardless of normalization choice in the conv body.
    """
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 32,
                 norm: str = "batchnorm",
                 raw_means_bypass: bool = False,
                 activation: str = "relu"):
        super().__init__()
        assert norm in ("batchnorm", "groupnorm"), norm
        assert activation in ("relu", "gelu"), activation
        self.in_channels = in_channels
        self.raw_means_bypass = raw_means_bypass

        def _norm(c: int) -> nn.Module:
            if norm == "groupnorm":
                # 4 groups divides 8, 16, 32 evenly
                return nn.GroupNorm(num_groups=4, num_channels=c)
            return nn.BatchNorm3d(c)

        def _act() -> nn.Module:
            return nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            # Input: (in_channels, 16, 32, 32)
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            _norm(8),
            _act(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (8, 8, 16, 16)

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            _norm(16),
            _act(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (16, 4, 8, 8)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            _norm(32),
            _act(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (32, 2, 4, 4)

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            _norm(32),
            _act(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
            # (32, 1, 1, 1)
        )

        # Optional bypass: 16-d projection of raw per-channel means.
        if self.raw_means_bypass:
            self.raw_proj = nn.Linear(in_channels, 16)
            mlp_in = 32 + 1 + 16
        else:
            mlp_in = 32 + 1

        # CNN flattened output (32) + Euclidean distance (1) [+ raw_means 16] -> latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, slabs: torch.Tensor, coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slabs: (B, C, Z, X, Y)
            coords_a: (B, 3)
            coords_b: (B, 3)
        Returns:
            edge_features: (B, latent_dim)
        """
        B = slabs.shape[0]

        # 1. Volumetric Feature Extraction
        cnn_out = self.features(slabs)  # (B, 32, 1, 1, 1)
        cnn_flat = cnn_out.view(B, -1)   # (B, 32)

        # 2. Distance Injection
        delta_s = torch.sqrt(torch.sum((coords_a - coords_b)**2, dim=1, keepdim=True))  # (B, 1)

        parts = [cnn_flat, delta_s]
        if self.raw_means_bypass:
            raw_means = slabs.mean(dim=(2, 3, 4))         # (B, in_channels)
            parts.append(self.raw_proj(raw_means))         # (B, 16)

        edge_features = self.mlp(torch.cat(parts, dim=1))  # (B, latent_dim)
        return edge_features


class ContinuousNodeCropper(nn.Module):
    """Single-well variant of ContinuousCropper.

    Extracts a (Z_out, 2P+1, 2P+1) box around a well's (x, y).
    XY span is the well center +/- pad. Z span is either:
      - "perforation": [perf_top, perf_bot]  (length varies per well; resampled to Z_out)
      - "full":        [0, Z_len-1]           (consistent across wells; Z_out cells = full reservoir)
    """

    def __init__(self, pad: int = 3, z_out: int = 16, z_extent: str = "full"):
        super().__init__()
        assert z_extent in ("perforation", "full"), z_extent
        self.pad = pad
        self.z_out = z_out
        self.z_extent = z_extent
        self.out_shape = (z_out, 2 * pad + 1, 2 * pad + 1)

    def forward(
        self,
        volumes: torch.Tensor,
        well_xyz: torch.Tensor,
        perf_range: torch.Tensor,
        full_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Args:
            volumes:     (B, C, Z, X, Y) — physics tensor expanded to batch size N_wells
            well_xyz:    (B, 3) continuous (x, y, z) of each well
            perf_range:  (B, 2) integer (perf_top, perf_bot) per well
            full_shape:  (Z_len, X_len, Y_len)
        Returns:
            slab: (B, C, Z_out, 2P+1, 2P+1)
        """
        B, C, _, _, _ = volumes.shape
        Z_len, X_len, Y_len = full_shape
        dtype, device = volumes.dtype, volumes.device

        well_x = well_xyz[:, 0]
        well_y = well_xyz[:, 1]

        pad = float(self.pad)
        x_min = well_x - pad
        x_max = well_x + pad
        y_min = well_y - pad
        y_max = well_y + pad
        if self.z_extent == "full":
            # Reservoir-wide Z — same physical resolution per slab voxel for every well.
            z_min = torch.zeros(B, dtype=dtype, device=device)
            z_max = torch.full((B,), float(Z_len - 1), dtype=dtype, device=device)
        else:
            z_min = perf_range[:, 0].to(dtype)
            z_max = perf_range[:, 1].to(dtype)
            # Ensure non-degenerate Z extent (e.g., n_layers == 1).
            z_max = torch.where(z_max - z_min < 1.0, z_min + 1.0, z_max)

        x_min_norm = (x_min / (X_len - 1)) * 2 - 1
        x_max_norm = (x_max / (X_len - 1)) * 2 - 1
        y_min_norm = (y_min / (Y_len - 1)) * 2 - 1
        y_max_norm = (y_max / (Y_len - 1)) * 2 - 1
        z_min_norm = (z_min / (Z_len - 1)) * 2 - 1
        z_max_norm = (z_max / (Z_len - 1)) * 2 - 1

        theta = torch.zeros((B, 3, 4), dtype=dtype, device=device)
        # Same axis convention as ContinuousCropper:
        #   row 0 -> grid_sample W (Y), row 1 -> grid_sample V (X), row 2 -> grid_sample U (Z)
        theta[:, 0, 0] = (y_max_norm - y_min_norm) / 2
        theta[:, 0, 3] = (y_max_norm + y_min_norm) / 2
        theta[:, 1, 1] = (x_max_norm - x_min_norm) / 2
        theta[:, 1, 3] = (x_max_norm + x_min_norm) / 2
        theta[:, 2, 2] = (z_max_norm - z_min_norm) / 2
        theta[:, 2, 3] = (z_max_norm + z_min_norm) / 2

        grid = F.affine_grid(theta, size=(B, C, *self.out_shape), align_corners=True)
        return F.grid_sample(volumes, grid, padding_mode="zeros", align_corners=True)


class PhysicsNodeSlabExtractor(nn.Module):
    """Assembles a node-slab input volume per well.

    Channels (in order):
      1..N_active: the named physics channels (e.g. PermX/Y/Z, Porosity, T0, P0, valid_mask)
      last:        perforation mask — 1 along Z inside [perf_top, perf_bot] (in slab
                   Z coordinates), 0 outside, broadcast across X/Y of the slab.

    For z_extent="full" (default), the slab Z axis spans the full active reservoir
    [0, Z_len-1] resampled to Z_out cells; the perforation mask carries the per-well
    info. For z_extent="perforation", the slab Z axis is exactly the well's perforation
    and the mask is 1 everywhere by construction.
    """

    def __init__(
        self,
        active_channels: list[str],
        pad: int = 3,
        z_out: int = 16,
        z_extent: str = "full",
    ):
        super().__init__()
        self.active_channels = active_channels
        self.pad = pad
        self.z_out = z_out
        self.z_extent = z_extent
        self.cropper = ContinuousNodeCropper(pad=pad, z_out=z_out, z_extent=z_extent)
        self.in_channels = len(active_channels) + 1  # + perforation mask

    @property
    def out_shape(self) -> tuple[int, int, int]:
        return self.cropper.out_shape

    def _build_perf_mask(
        self, perf_range: torch.Tensor, full_shape: tuple[int, int, int], dtype, device
    ) -> torch.Tensor:
        """Per-voxel perforation indicator at slab Z resolution.

        For "full" z_extent the slab Z axis maps voxel z' in [0, Z_out-1] to reservoir
        Z in [0, Z_len-1]; we mark voxels whose underlying Z falls inside [perf_top, perf_bot].
        For "perforation" z_extent every slab voxel is perforated by construction.
        """
        B = perf_range.shape[0]
        Z_out, X_out, Y_out = self.out_shape
        if self.z_extent == "perforation":
            return torch.ones((B, 1, Z_out, X_out, Y_out), dtype=dtype, device=device)
        # z_extent == "full"
        Z_len = full_shape[0]
        # Slab Z voxel centers in reservoir Z coords (align_corners=True semantics).
        z_axis = torch.linspace(0.0, float(Z_len - 1), Z_out, dtype=dtype, device=device)
        perf_top = perf_range[:, 0].to(dtype).view(B, 1)
        perf_bot = perf_range[:, 1].to(dtype).view(B, 1)
        inside = ((z_axis.view(1, Z_out) >= perf_top) & (z_axis.view(1, Z_out) <= perf_bot)).to(dtype)
        # Broadcast over X/Y
        return inside.view(B, 1, Z_out, 1, 1).expand(B, 1, Z_out, X_out, Y_out)

    def forward(
        self,
        volumes_dict: dict[str, torch.Tensor],
        well_xyz: torch.Tensor,
        perf_range: torch.Tensor,
        full_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        device = well_xyz.device
        tensors = []
        for ch in self.active_channels:
            t = volumes_dict[ch].to(device, non_blocking=True)
            if t.ndim == 4:  # (B, Z, X, Y) -> (B, 1, Z, X, Y)
                t = t.unsqueeze(1)
            tensors.append(t)
        multi_vol = torch.cat(tensors, dim=1)  # (B, C_phys, Z, X, Y)

        slab = self.cropper(multi_vol, well_xyz, perf_range, full_shape)
        perf_mask = self._build_perf_mask(perf_range, full_shape, slab.dtype, slab.device)
        return torch.cat([slab, perf_mask], dim=1)


class PhysicsNodeSlabCNN(nn.Module):
    """Small 3D CNN for per-node physics slabs.

    Input shape: (B, in_channels, Z, 2P+1, 2P+1). For P=3 the XY extent is 7.

    Design choices and the failure modes they target:
      - **No XY pooling.** XY extent (7 at P=3) is small enough that pooling
        destroys ~3/4 of lateral context. Conv3d keeps XY through every block.
      - **Z-strided convolutions, no Z-pool.** Z is the long axis (16);
        stride-2 conv halves it per block.
      - **GroupNorm, NOT BatchNorm.** Diagnostic on the v2 model (BatchNorm3d
        based) showed that BN strips absolute-scale signal across instances:
        the CNN learned local-texture features but couldn't encode geology
        identity (only 5.7% between-geology variance vs 22% for the hand-
        engineered profile). GroupNorm normalizes within an instance only,
        preserving inter-instance scale differences.
      - **Raw per-channel slab means concatenated to the MLP.** Bypasses ALL
        normalization for absolute-magnitude signal. Replicates the strongest
        signal in the hand-engineered profile (channel-wise means over the
        well column) while letting the CNN body learn spatial structure on top.
      - **AdaptiveAvg + AdaptiveMax** at the final pool to preserve both
        first-moment and extreme-value statistics from the spatial features.
    """

    def __init__(self, in_channels: int, latent_dim: int = 32, activation: str = "relu"):
        super().__init__()
        assert activation in ("relu", "gelu"), activation
        self.in_channels = in_channels
        # GroupNorm groups: use 4 to avoid edge cases with non-power-of-2 channels.
        ng = 4

        def _act() -> nn.Module:
            return nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=ng, num_channels=16),
            _act(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=(2, 1, 1)),
            nn.GroupNorm(num_groups=ng, num_channels=32),
            _act(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=(2, 1, 1)),
            nn.GroupNorm(num_groups=ng, num_channels=32),
            _act(),
        )
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        # Raw-channel-mean bypass: projects per-channel slab means into a
        # supplementary feature without ever passing them through a norm layer.
        self.raw_proj = nn.Linear(in_channels, 16)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 32 + 16, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, slabs: torch.Tensor) -> torch.Tensor:
        B = slabs.shape[0]
        # Raw per-channel means over (Z, X, Y) — absolute-scale signal that
        # the GroupNorm'd CNN body would otherwise lose at training time.
        raw_means = slabs.mean(dim=(2, 3, 4))  # (B, in_channels)
        raw_feat = self.raw_proj(raw_means)    # (B, 16)
        feat = self.features(slabs)
        avg = self.avg_pool(feat).view(B, -1)  # (B, 32)
        mx = self.max_pool(feat).view(B, -1)   # (B, 32)
        return self.mlp(torch.cat([avg, mx, raw_feat], dim=1))


class PhysicsSlabSVD(nn.Module):
    """
    SVD-based Edge Feature Extractor.
    Takes [B, C, Z=16, X=32, Y=32] volumetric slabs, flattens them,
    projects using precomputed SVD weights, injects delta_s (continuous Euclidean distance),
    and outputs a configured latent edge dimension.
    """
    def __init__(self, 
                 svd_weights_path: str,
                 latent_dim: int = 32,
                 k: int = 32):
        super().__init__()
        
        try:
            state = torch.load(svd_weights_path, weights_only=True)
            if isinstance(state, torch.Tensor):
                components = state
                mean = torch.zeros(components.shape[1], dtype=torch.float32)
                std = torch.ones(components.shape[1], dtype=torch.float32)
            else:
                components = state["components"]
                mean = state["mean"]
                std = state["std"]
        except Exception as e:
            raise ValueError(f"Failed to load SVD weights from {svd_weights_path}: {e}")
            
        M = components.shape[1]
        self.register_buffer("voxel_mean", mean)
        self.register_buffer("voxel_std", std)
        
        self.svd_proj = nn.Linear(M, k, bias=False)
        self.svd_proj.weight.data = components
        self.svd_proj.weight.requires_grad = False
        
        # SVD projected output (k) + Euclidean distance (1) -> latent_dim
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(k + 1),
            nn.Linear(k + 1, 32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, slabs: torch.Tensor, coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slabs: (B, C, Z, X, Y)
            coords_a: (B, 3) 
            coords_b: (B, 3)
        Returns:
            edge_features: (B, latent_dim)
        """
        B = slabs.shape[0]
        
        # 1. Flatten the entire multidimensional tensor except batch
        flat_slabs = slabs.view(B, -1)   # (B, M)
        
        # 1.5 Standardize exactly sequentially with SVD precomputation scale
        flat_slabs = (flat_slabs - self.voxel_mean) / self.voxel_std
        
        # 2. Linear projection using SVD base
        svd_emb = self.svd_proj(flat_slabs) # (B, k)
        
        # 3. Distance Injection
        delta_s = torch.sqrt(torch.sum((coords_a - coords_b)**2, dim=1, keepdim=True))  # (B, 1)
        
        # 4. Concatenate and Compress
        concat_vec = torch.cat([svd_emb, delta_s], dim=1)
        edge_features = self.mlp(concat_vec)  # (B, latent_dim)
        
        return edge_features
