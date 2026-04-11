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
    """
    def __init__(self, 
                 in_channels: int, 
                 latent_dim: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: (in_channels, 16, 32, 32)
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (8, 8, 16, 16)
            
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (16, 4, 8, 8)
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # (32, 2, 4, 4)
            
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
            # (32, 1, 1, 1)
        )
        
        # CNN flattened output (32) + Euclidean distance (1) -> latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(32 + 1, 32),
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
        cnn_out = self.features(slabs)  # (B, 128, 1, 1, 1)
        cnn_flat = cnn_out.view(B, -1)   # (B, 128)
        
        # 2. Distance Injection
        delta_s = torch.sqrt(torch.sum((coords_a - coords_b)**2, dim=1, keepdim=True))  # (B, 1)
        
        # 3. Concatenate and Compress
        concat_vec = torch.cat([cnn_flat, delta_s], dim=1)
        edge_features = self.mlp(concat_vec)  # (B, latent_dim)
        
        return edge_features


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
