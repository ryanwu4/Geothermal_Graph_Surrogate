import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path('data_test')
h5_files = sorted([f for f in data_dir.glob('*.h5') if f.name != 'minimal_compiled.h5' and f.name != 'minimal_compiled_with_wept.h5'])

# Select 3 files to compare
files_to_compare = h5_files[:3]

# Find where the wells actually are to pick better Z levels
with h5py.File(files_to_compare[0], 'r') as f:
    is_well = f['Input/IsWell'][...]
    z_indices = np.where(is_well == 1)[0]
    if len(z_indices) > 0:
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        z_levels = [z_min, (z_min + z_max) // 2, z_max]
    else:
        z_levels = [10, 20, 30] # fallback

print(f"Plotting Z levels where wells are located: {z_levels}")

vars_to_plot = ['Input/PermX', 'Input/Porosity', 'Input/Temperature0', 'Input/Pressure0']

fig, axes = plt.subplots(len(vars_to_plot) * len(z_levels), len(files_to_compare), 
                         figsize=(5 * len(files_to_compare), 4 * len(vars_to_plot) * len(z_levels)))

for i, var in enumerate(vars_to_plot):
    for j, z in enumerate(z_levels):
        row_idx = i * len(z_levels) + j
        
        # Find global min/max for consistent colorbars across files
        vmin, vmax = float('inf'), float('-inf')
        for f_path in files_to_compare:
            with h5py.File(f_path, 'r') as f:
                data_slice = f[var][z, :, :]
                valid_mask = data_slice != -999
                if np.any(valid_mask):
                    vmin = min(vmin, np.min(data_slice[valid_mask]))
                    vmax = max(vmax, np.max(data_slice[valid_mask]))
                
        for k, f_path in enumerate(files_to_compare):
            ax = axes[row_idx, k]
            with h5py.File(f_path, 'r') as f:
                data_slice = f[var][z, :, :]
                
                # Mask out -999 values
                data_masked = np.ma.masked_where(data_slice == -999, data_slice)
                
                # Use log scale for permeability if it varies widely
                if 'Perm' in var and vmax > 0 and vmin > 0 and vmax / vmin > 100:
                    im = ax.imshow(np.log10(data_masked), cmap='viridis')
                    title_var = f"log10({var.split('/')[-1]})"
                else:
                    im = ax.imshow(data_masked, cmap='viridis', vmin=vmin, vmax=vmax)
                    title_var = var.split('/')[-1]
                
                ax.set_title(f"{f_path.name}\n{title_var} at Z={z}")
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('rock_properties_comparison_well_depths.png', dpi=150, bbox_inches='tight')
print("Saved plot to rock_properties_comparison_well_depths.png")

print("\n--- Scalar Parameters (Input/ParamsScalar) ---")
for f_path in files_to_compare:
    with h5py.File(f_path, 'r') as f:
        params = f['Input/ParamsScalar'][...]
        print(f"{f_path.name}: {params}")
