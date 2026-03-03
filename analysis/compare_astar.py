import sys
from pathlib import Path
import argparse

import h5py
import numpy as np
import plotly.graph_objects as go
import heapq

# Import the new A* algorithm
from geothermal.geology_graph import generate_geology_edges as generate_new_edges

# ======================================================================
# COPY OF OLD A* IMPLEMENTATION FOR BASELINE COMPARISON
# ======================================================================
def generate_old_edges(
    perm_grid: np.ndarray,
    porosity_grid: np.ndarray,
    temp_grid: np.ndarray,
    press_grid: np.ndarray,
    is_injector: np.ndarray,
    well_coords: np.ndarray,
    k_neighbors: int = 2,
    return_paths: bool = False,
) -> tuple:
    num_wells = well_coords.shape[0]
    if num_wells < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 14), dtype=np.float32)

    well_zxy = np.stack(
        [well_coords[:, 2], well_coords[:, 0], well_coords[:, 1]], axis=1
    )

    max_z, max_x, max_y = perm_grid.shape
    min_perm_threshold = 1e-15
    valid_mask = (perm_grid > min_perm_threshold) & (porosity_grid > 0.0)

    max_perm_val = max(float(np.max(perm_grid)), 1e-12)
    min_poro_val = (
        min(float(np.min(porosity_grid[valid_mask])), 1.0)
        if np.any(valid_mask)
        else 1e-4
    )

    neighbors = [
        (1, 0, 0, 1.0),
        (-1, 0, 0, 1.0),
        (0, 1, 0, 1.0),
        (0, -1, 0, 1.0),
        (0, 0, 1, 1.0),
        (0, 0, -1, 1.0),
    ]

    def _harmonic_mean(a: float, b: float) -> float:
        if a <= 0 or b <= 0:
            return 0.0
        return 2.0 * a * b / (a + b)

    def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        dist = abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
        return dist * (min_poro_val / max_perm_val)

    edge_index_list = []
    edge_attr_list = []
    path_coords_list = []

    target_goals = []
    for start_idx in range(num_wells):
        dists = np.sqrt(np.sum((well_coords[start_idx] - well_coords) ** 2, axis=1))

        inj_mask = is_injector & (np.arange(num_wells) != start_idx)
        ext_mask = (~is_injector) & (np.arange(num_wells) != start_idx)

        inj_indices = np.where(inj_mask)[0]
        ext_indices = np.where(ext_mask)[0]

        if len(inj_indices) > 0:
            closest_injs = inj_indices[np.argsort(dists[inj_indices])[:k_neighbors]]
            for goal_idx in closest_injs:
                target_goals.append((start_idx, goal_idx))

        if len(ext_indices) > 0:
            closest_exts = ext_indices[np.argsort(dists[ext_indices])[:k_neighbors]]
            for goal_idx in closest_exts:
                target_goals.append((start_idx, goal_idx))

    for start_idx, goal_idx in target_goals:
        start_pos = tuple(well_zxy[start_idx])
        goal_pos = tuple(well_zxy[goal_idx])

        if not valid_mask[start_pos] or not valid_mask[goal_pos]:
            continue

        open_set = []
        heapq.heappush(open_set, (0.0, start_pos))

        cost_so_far = {start_pos: 0.0}
        came_from = {start_pos: None}
        path_length = {start_pos: 0}
        sum_inv_perm = {start_pos: 1.0 / max(float(perm_grid[start_pos]), 1e-30)}
        sum_inv_poro = {start_pos: 1.0 / max(float(porosity_grid[start_pos]), 1e-30)}

        found_path = False

        while open_set:
            current_priority, current_pos = heapq.heappop(open_set)

            if current_pos == goal_pos:
                found_path = True
                break

            cz, cx, cy = current_pos
            c_perm = perm_grid[cz, cx, cy]
            c_poro = porosity_grid[cz, cx, cy]

            for dz, dx, dy, dist in neighbors:
                nz, nx, ny = cz + dz, cx + dx, cy + dy

                if not (0 <= nz < max_z and 0 <= nx < max_x and 0 <= ny < max_y):
                    continue

                if not valid_mask[nz, nx, ny]:
                    continue

                n_pos = (nz, nx, ny)
                n_perm = perm_grid[nz, nx, ny]
                n_poro = porosity_grid[nz, nx, ny]

                hm_perm_edge = _harmonic_mean(c_perm, n_perm)
                hm_poro_edge = _harmonic_mean(c_poro, n_poro)

                if hm_perm_edge == 0.0:
                    continue

                step_cost = (hm_poro_edge / hm_perm_edge) * dist
                new_cost = cost_so_far[current_pos] + step_cost

                if n_pos not in cost_so_far or new_cost < cost_so_far[n_pos]:
                    cost_so_far[n_pos] = new_cost
                    priority = new_cost + _heuristic(n_pos, goal_pos)
                    heapq.heappush(open_set, (priority, n_pos))
                    came_from[n_pos] = current_pos

                    path_length[n_pos] = path_length[current_pos] + 1
                    sum_inv_perm[n_pos] = sum_inv_perm[current_pos] + 1.0 / max(float(n_perm), 1e-30)
                    sum_inv_poro[n_pos] = sum_inv_poro[current_pos] + 1.0 / max(float(n_poro), 1e-30)

        if found_path:
            plen = path_length[goal_pos]
            t_cost = cost_so_far[goal_pos]
            edge_index_list.append([start_idx, goal_idx])
            edge_attr_list.append([plen, t_cost] + [0]*12)  # Pad for dummy
            
            if return_paths:
                cp = goal_pos
                p = []
                while cp is not None:
                    p.append(cp)
                    cp = came_from[cp]
                p.reverse()
                p_array = np.array(p)
                out_p = np.stack([p_array[:, 1], p_array[:, 2], p_array[:, 0]], axis=1)
                path_coords_list.append(out_p)

    if len(edge_index_list) == 0:
        if return_paths:
            return np.empty((2, 0), dtype=np.int64), np.empty((0, 14), dtype=np.float32), []
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 14), dtype=np.float32)

    if return_paths:
        return np.array(edge_index_list).T, np.array(edge_attr_list, dtype=np.float32), path_coords_list
    return np.array(edge_index_list).T, np.array(edge_attr_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new A* paths")
    parser.add_argument("--h5-path", type=Path, default="data/sample.h5", help="Path to a geothermal HDF5 file")
    parser.add_argument("--case-id", type=str, default=None, help="Specific case ID to use (default: first available)")
    args = parser.parse_args()

    if not args.h5_path.exists():
        print(f"Error: Could not find HDF5 file at {args.h5_path}")
        print("Please point to a valid HDF5 to test paths. Generating mock data for visualization...")
        
        # fallback to mock data if file not provided
        nz, nx, ny = 20, 20, 20
        px = np.ones((nz, nx, ny)) * 1.1e-15
        py = np.ones((nz, nx, ny)) * 1.1e-15
        pz = np.ones((nz, nx, ny)) * 1.1e-15
        # Corridors: High perm in X-Z diagonal
        for i in range(20):
            px[i, i, 10] = 1e-11
            py[i, i, 10] = 1e-11
            pz[i, i, 10] = 1e-11

        poro = np.ones((nz, nx, ny)) * 0.1
        t0 = np.ones((nz, nx, ny)) * 200.0  # 200 C
        p0 = np.ones((nz, nx, ny)) * 20e6   # 20 MPa
        is_inj = np.array([True, False])
        
        # Start at corner, go to other corner
        well_coords_old = np.array([[0, 10, 0], [19, 10, 19]])
        well_coords_new = np.zeros((2, 1, 3), dtype=np.int32)
        well_coords_new[:, 0, :] = well_coords_old
        
        perm_avg = (px + py + pz) / 3.0
    else:
        with h5py.File(args.h5_path, 'r') as f:
            g = f # Test file has Root structure directly

            is_well = g['Input/IsWell'][:]
            inj_rate = g['Input/InjRate'][:]
            px = g['Input/PermX'][:]
            py = g['Input/PermY'][:]
            pz = g['Input/PermZ'][:]
            poro = g['Input/Porosity'][:]
            t0 = g['Input/Temperature0'][:]
            p0 = g['Input/Pressure0'][:]
            
            well_mask = is_well == 1
            x_idx, y_idx = np.where(np.any(well_mask, axis=0))
            if len(x_idx) < 2:
                raise ValueError("Need at least 2 wells to test paths")
            
            # just pick 2 wells for demonstration
            x_idx, y_idx = x_idx[:2], y_idx[:2]
            
            # deepest z
            z_indices = np.arange(is_well.shape[0])[:, None, None]
            depth = np.where(well_mask, z_indices, -1).max(axis=0)[x_idx, y_idx]
            is_inj = inj_rate[depth, x_idx, y_idx] > 0
            
            well_coords_old = np.stack([x_idx, y_idx, depth], axis=1)
            well_coords_new = np.expand_dims(well_coords_old, axis=1)
            
            perm_avg = (px + py + pz) / 3.0

    print("Running OLD A* (Isotropic, 6-connected)...")
    idx_old, attr_old, paths_old = generate_old_edges(
        perm_avg, poro, t0, p0, is_inj, well_coords_old, k_neighbors=1, return_paths=True
    )
    
    print("Running NEW A* (Anisotropic, 26-connected, T-visc)...")
    idx_new, attr_new, paths_new = generate_new_edges(
        px, py, pz, poro, t0, p0, is_inj, well_coords_new, k_neighbors=1, return_paths=True
    )
    
    if len(paths_old) == 0 or len(paths_new) == 0:
        print("No paths found. Check grid connectivity.")
        return

    # Extract first path found
    p_old = paths_old[0]
    p_new = paths_new[0][0] # First connection, first inner path (out of N=1 paths)

    # Convert node indices back to geometric coords for plotting
    x_old, y_old, z_old = p_old[:, 0], p_old[:, 1], p_old[:, 2]
    x_new, y_new, z_new = p_new[:, 0], p_new[:, 1], p_new[:, 2]

    # Plotly integration
    fig = go.Figure()

    # Old Path
    fig.add_trace(go.Scatter3d(
        x=x_old, y=y_old, z=z_old,
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=4),
        name="Old A* (6-way, Iso)"
    ))

    # New Path
    fig.add_trace(go.Scatter3d(
        x=x_new, y=y_new, z=z_new,
        mode='lines+markers',
        line=dict(color='green', width=5),
        marker=dict(size=4),
        name="New A* (26-way, Aniso)"
    ))

    # Wells points
    xw, yw, zw = well_coords_old[:, 0], well_coords_old[:, 1], well_coords_old[:, 2]
    fig.add_trace(go.Scatter3d(
        x=xw, y=yw, z=zw,
        mode='markers',
        marker=dict(size=8, color=['blue' if inj else 'orange' for inj in is_inj], symbol='diamond'),
        name="Wells"
    ))

    fig.update_layout(
        title="Comparison of A* Pathfinding Algorithms",
        scene=dict(
            xaxis_title='X Index',
            yaxis_title='Y Index',
            zaxis_title='Z (Depth) Index',
            # Inverse Z axis since depth increases downwards
            zaxis=dict(autorange="reversed")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    out_html = "astar_comparison.html"
    fig.write_html(out_html)
    print(f"Interactive plot successfully saved to {out_html}. Open it in your browser!")

if __name__ == "__main__":
    main()
