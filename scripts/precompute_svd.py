#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from geothermal.data import load_hetero_graphs
from geothermal.physics_slab import PhysicsSlabExtractor
from geothermal.model import EDGE_TYPES, seed_all

def main():
    parser = argparse.ArgumentParser("Precompute SVD weights using randomized SVD algorithm.")
    parser.add_argument("--h5-path", type=Path, default=Path("data/compiled_full_CNN.h5"))
    parser.add_argument("--output-path", type=Path, default=Path("svd_components.pt"))
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    seed_all(args.seed)
    
    print(f"Loading graphs from {args.h5_path}...")
    start_load = time.time()
    try:
        graphs, _ = load_hetero_graphs(args.h5_path, target="graph_energy_total")
    except Exception as e:
        print(f"Failed to load dataset at {args.h5_path}: {e}")
        return
        
    print(f"Loaded {len(graphs)} graphs. Time taken: {time.time() - start_load:.2f}s")
    if len(graphs) == 0:
        print("No graphs found. Exiting.")
        return
    
    # Shuffle graphs to ensure randomized sampling
    np.random.shuffle(graphs)
    
    device = torch.device(args.device)
    active_channels = ["PermX", "PermY", "PermZ", "Porosity", "Temperature0", "Pressure0", "valid_mask"]
    extractor = PhysicsSlabExtractor(active_channels=active_channels, out_shape=(16, 32, 32)).to(device)
    
    buffer = []
    total_samples = 0
    
    print("Beginning continuous physical slab extractions...")
    start_extract = time.time()
    
    for graph in tqdm(graphs, desc="Extracting all slabs"):
        graph = graph.to(device)
        for k, v in graph.physics_context.d.items():
            graph.physics_context.d[k] = v.to(device)
            
        phys_dict = graph.physics_context.d
        full_shape = graph.physics_context.full_shape
        
        for edge_type in EDGE_TYPES:
            edge_index = graph[edge_type].edge_index
            num_edges = edge_index.shape[1]
            if num_edges == 0:
                continue
                
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            coords_a = graph["well"].pos_xyz[src_nodes]
            coords_b = graph["well"].pos_xyz[dst_nodes]
            
            # Expand physics dict
            phys_expanded = {}
            for pk, pv in phys_dict.items():
                phys_expanded[pk] = pv.unsqueeze(0).expand(num_edges, -1, -1, -1)
                
            # Extract
            with torch.no_grad():
                slabs = extractor(phys_expanded, coords_a, coords_b, full_shape)
                # Keep flat slabs on CPU to prevent VRAM overflow over thousands of samples
                flat_slabs = slabs.view(num_edges, -1).cpu() 
                
            buffer.append(flat_slabs)
            total_samples += num_edges
            
    if len(buffer) == 0:
        print("No edges found in the dataset.")
        return
        
    print(f"Collected {total_samples} edge slabs in {time.time() - start_extract:.2f}s. Stacking matrix...")
    start_stack = time.time()
    X = torch.cat(buffer, dim=0)  # Shape: (N, M)
    print(f"Matrix stacked in {time.time() - start_stack:.2f}s.")
    
    print(f"Running voxel-wise standard scaling on matrix of shape {X.shape}...")
    start_scale = time.time()
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X_std[X_std < 1e-8] = 1.0  # Prevent zero divisions for constantly zero voxels
    
    X_scaled = (X - X_mean) / X_std
    print(f"Standard scaling completed in {time.time() - start_scale:.2f}s.")
    
    print(f"Running randomized low-rank PCA on scaled matrix...")
    start_pca = time.time()
    # Compute PCA on CPU; center=False since we already manual centered it
    U, S, V = torch.pca_lowrank(X_scaled, q=args.k, center=False)
    print(f"PCA completed in {time.time() - start_pca:.2f}s.")
    
    # V shape is (M, k). The neural network projection matrix expects (k, M).
    components_tensor = V.t()
    
    print(f"Finished fitting PCA. Saving scaled components to {args.output_path}")
    state = {
        "components": components_tensor,
        "mean": X_mean.squeeze(0),
        "std": X_std.squeeze(0)
    }
    torch.save(state, args.output_path)
    print("Done!")

if __name__ == "__main__":
    main()
