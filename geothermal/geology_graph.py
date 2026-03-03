"""A* search-based geology-aware graph construction for geothermal well networks."""

from __future__ import annotations

import numpy as np
import heapq
import cantera as ct

def get_water_viscosity(T_C: float, P_Pa: float, w=None) -> float:
    """Get the dynamic viscosity of liquid water (Pa*s)."""
    if w is None:
        try:
            w = ct.Water()
        except:
            return 2.4e-5 * 10**(247.8 / (T_C + 273.15 - 140))
    try:
        w.TP = T_C + 273.15, max(P_Pa, 1000.0)
        return w.viscosity
    except Exception:
        return 2.4e-5 * 10**(247.8 / (T_C + 273.15 - 140))

def generate_geology_edges(
    perm_x_grid: np.ndarray,
    perm_y_grid: np.ndarray,
    perm_z_grid: np.ndarray,
    porosity_grid: np.ndarray,
    temp_grid: np.ndarray,
    press_grid: np.ndarray,
    is_injector: np.ndarray,
    well_coords: np.ndarray,
    k_neighbors: int = 2,
    return_paths: bool = False,
) -> tuple:
    """
    Computes geological paths between valid well pairs using the A* algorithm.
    Allows for anisotropic permeability and multiple paths per well-pair.
    """
    num_wells = well_coords.shape[0]
    if num_wells < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 18), dtype=np.float32)

    if well_coords.ndim == 2:
        well_coords = np.expand_dims(well_coords, axis=1)
    
    num_paths = well_coords.shape[1]

    # Convert coordinates to Z, X, Y format
    well_zxy = np.stack(
        [well_coords[:, :, 2], well_coords[:, :, 0], well_coords[:, :, 1]], axis=2
    )

    max_z, max_x, max_y = perm_x_grid.shape

    min_perm_threshold = 1e-15
    perm_mag = (perm_x_grid + perm_y_grid + perm_z_grid) / 3.0
    valid_mask = (perm_mag > min_perm_threshold) & (porosity_grid > 0.0)

    max_perm_val = max(float(np.max(perm_mag)), 1e-12)
    min_poro_val = (
        min(float(np.min(porosity_grid[valid_mask])), 1.0)
        if np.any(valid_mask)
        else 1e-4
    )

    # 3D 26-way neighborhood
    neighbors = []
    for dz in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dz == 0 and dx == 0 and dy == 0:
                    continue
                dist = np.sqrt(dz*dz + dx*dx + dy*dy)
                neighbors.append((dz, dx, dy, dist, dz/dist, dx/dist, dy/dist))

    def _harmonic_mean(a: float, b: float) -> float:
        if a <= 0 or b <= 0:
            return 0.0
        return 2.0 * a * b / (a + b)

    def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        dist = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
        return dist * (min_poro_val * 1e-4 / max_perm_val)
        
    try:
        w_ct = ct.Water()
    except:
        w_ct = None

    # Pre-compute viscosity for all valid cells to prevent massive Cantera overhead inside A* inner loop
    visc_grid = np.zeros_like(temp_grid)
    valid_z, valid_x, valid_y = np.where(valid_mask)
    for vz, vx, vy in zip(valid_z, valid_x, valid_y):
        t = temp_grid[vz, vx, vy]
        p = press_grid[vz, vx, vy]
        t_eff = t if t > -900 else 150.0
        p_eff = p if p > -900 else 20e6
        visc_grid[vz, vx, vy] = get_water_viscosity(t_eff, p_eff, w_ct)

    edge_index_list = []
    edge_attr_list = []
    path_coords_list = []

    well_centers = well_coords[:, num_paths // 2, :]
    
    target_goals = []
    for start_idx in range(num_wells):
        dists = np.sqrt(np.sum((well_centers[start_idx] - well_centers) ** 2, axis=1))

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
        succ_attr = []
        succ_paths = []
        
        for path_idx in range(num_paths):
            start_pos = tuple(well_zxy[start_idx, path_idx])
            goal_pos = tuple(well_zxy[goal_idx, path_idx])

            if not valid_mask[start_pos] or not valid_mask[goal_pos]:
                continue

            open_set = []
            heapq.heappush(open_set, (0.0, start_pos))

            cost_so_far = {start_pos: 0.0}
            came_from = {start_pos: None}
            path_length = {start_pos: 0}
            
            c_iso_perm = (perm_x_grid[start_pos] + perm_y_grid[start_pos] + perm_z_grid[start_pos])/3.0
            min_perm = {start_pos: c_iso_perm}
            max_perm = {start_pos: c_iso_perm}
            sum_inv_perm = {start_pos: 1.0 / max(float(c_iso_perm), 1e-30)}

            min_poro = {start_pos: porosity_grid[start_pos]}
            max_poro = {start_pos: porosity_grid[start_pos]}
            sum_inv_poro = {start_pos: 1.0 / max(float(porosity_grid[start_pos]), 1e-30)}

            t_start = temp_grid[start_pos]
            t_start = t_start if t_start > -900 else 0.0
            p_start = press_grid[start_pos]
            p_start = p_start if p_start > -900 else 0.0

            min_temp = {start_pos: t_start}
            max_temp = {start_pos: t_start}
            min_press = {start_pos: p_start}
            max_press = {start_pos: p_start}
            
            sum_inv_aniso_perm = {start_pos: 0.0}
            
            c_visc = visc_grid[start_pos]
            sum_inv_visc = {start_pos: 1.0 / max(c_visc, 1e-30)}

            found_path = False

            while open_set:
                current_priority, current_pos = heapq.heappop(open_set)

                if current_pos == goal_pos:
                    found_path = True
                    break

                cz, cx, cy = current_pos
                c_poro = porosity_grid[cz, cx, cy]
                
                c_viscosity = visc_grid[cz, cx, cy]

                for dz, dx, dy, dist, nz, nx, ny in neighbors:
                    nz_idx, nx_idx, ny_idx = cz + dz, cx + dx, cy + dy

                    if not (0 <= nz_idx < max_z and 0 <= nx_idx < max_x and 0 <= ny_idx < max_y):
                        continue

                    if not valid_mask[nz_idx, nx_idx, ny_idx]:
                        continue

                    n_pos = (nz_idx, nx_idx, ny_idx)
                    n_poro = porosity_grid[n_pos]
                    
                    k_c_aniso = perm_x_grid[current_pos] * (nx**2) + perm_y_grid[current_pos] * (ny**2) + perm_z_grid[current_pos] * (nz**2)
                    k_n_aniso = perm_x_grid[n_pos] * (nx**2) + perm_y_grid[n_pos] * (ny**2) + perm_z_grid[n_pos] * (nz**2)

                    n_viscosity = visc_grid[nz_idx, nx_idx, ny_idx]

                    hm_perm_edge = _harmonic_mean(k_c_aniso, k_n_aniso)
                    hm_poro_edge = _harmonic_mean(c_poro, n_poro)
                    hm_visc_edge = _harmonic_mean(c_viscosity, n_viscosity)

                    if hm_perm_edge == 0.0:
                        continue

                    step_cost = (hm_poro_edge * hm_visc_edge / hm_perm_edge) * dist
                    new_cost = cost_so_far[current_pos] + step_cost

                    if n_pos not in cost_so_far or new_cost < cost_so_far[n_pos]:
                        cost_so_far[n_pos] = new_cost
                        priority = new_cost + _heuristic(n_pos, goal_pos)
                        heapq.heappush(open_set, (priority, n_pos))
                        came_from[n_pos] = current_pos

                        path_length[n_pos] = path_length[current_pos] + 1
                        
                        n_iso_perm = (perm_x_grid[n_pos] + perm_y_grid[n_pos] + perm_z_grid[n_pos])/3.0
                        min_perm[n_pos] = min(min_perm[current_pos], n_iso_perm)
                        max_perm[n_pos] = max(max_perm[current_pos], n_iso_perm)
                        sum_inv_perm[n_pos] = sum_inv_perm[current_pos] + 1.0 / max(float(n_iso_perm), 1e-30)

                        min_poro[n_pos] = min(min_poro[current_pos], n_poro)
                        max_poro[n_pos] = max(max_poro[current_pos], n_poro)
                        sum_inv_poro[n_pos] = sum_inv_poro[current_pos] + 1.0 / max(float(n_poro), 1e-30)
                        
                        sum_inv_aniso_perm[n_pos] = sum_inv_aniso_perm[current_pos] + 1.0 / max(float(hm_perm_edge), 1e-30)
                        sum_inv_visc[n_pos] = sum_inv_visc[current_pos] + 1.0 / max(float(n_viscosity), 1e-30)

                        n_temp = temp_grid[n_pos]
                        n_press = press_grid[n_pos]
                        n_temp = n_temp if n_temp > -900 else min_temp[current_pos]
                        n_press = n_press if n_press > -900 else min_press[current_pos]

                        min_temp[n_pos] = min(min_temp[current_pos], n_temp)
                        max_temp[n_pos] = max(max_temp[current_pos], n_temp)
                        min_press[n_pos] = min(min_press[current_pos], n_press)
                        max_press[n_pos] = max(max_press[current_pos], n_press)

            if found_path:
                plen = path_length[goal_pos]
                n_nodes = plen + 1

                path_hm_perm = n_nodes / sum_inv_perm[goal_pos]
                m_perm = min_perm[goal_pos]
                mx_perm = max_perm[goal_pos]

                path_hm_poro = n_nodes / sum_inv_poro[goal_pos]
                m_poro = min_poro[goal_pos]
                mx_poro = max_poro[goal_pos]

                m_t = min_temp[goal_pos]
                m_p = min_press[goal_pos]
                
                path_hm_aniso_perm = plen / max(1e-30, sum_inv_aniso_perm[goal_pos]) if plen > 0 else path_hm_perm
                path_hm_visc = n_nodes / sum_inv_visc[goal_pos]

                t_goal_raw = temp_grid[goal_pos]
                p_goal_raw = press_grid[goal_pos]
                t_goal = t_goal_raw if t_goal_raw > -900 else t_start
                p_goal = p_goal_raw if p_goal_raw > -900 else p_start

                delta_t = t_goal - t_start
                delta_p = p_goal - p_start

                grad_t = delta_t / max(plen, 1)
                grad_p = delta_p / max(plen, 1)

                t_cost = cost_so_far[goal_pos]

                succ_attr.append(
                    [
                        plen,
                        t_cost,
                        m_perm,
                        mx_perm,
                        path_hm_perm,
                        m_poro,
                        mx_poro,
                        path_hm_poro,
                        delta_t,
                        delta_p,
                        grad_t,
                        grad_p,
                        m_t,
                        m_p,
                        path_hm_visc,
                        path_hm_aniso_perm,
                    ]
                )

                if return_paths:
                    cp = goal_pos
                    p = []
                    while cp is not None:
                        p.append(cp)
                        cp = came_from[cp]
                    p.reverse()
                    p_array = np.array(p)
                    out_p = np.stack([p_array[:, 1], p_array[:, 2], p_array[:, 0]], axis=1)
                    succ_paths.append(out_p)

        if len(succ_attr) > 0:
            arr = np.array(succ_attr, dtype=np.float32)
            
            avg_attr = np.zeros(18, dtype=np.float32)
            avg_attr[0] = np.mean(arr[:, 0]) # plen
            avg_attr[1] = np.mean(arr[:, 1]) # t_cost (mean_tof)
            avg_attr[2] = np.min(arr[:, 2])  # m_perm
            avg_attr[3] = np.max(arr[:, 3])  # mx_perm
            avg_attr[4] = np.mean(arr[:, 4]) # hm_perm
            avg_attr[5] = np.min(arr[:, 5])  # m_poro
            avg_attr[6] = np.max(arr[:, 6])  # mx_poro
            avg_attr[7] = np.mean(arr[:, 7]) # hm_poro
            avg_attr[8] = np.mean(arr[:, 8]) # delta_t
            avg_attr[9] = np.mean(arr[:, 9]) # delta_p
            avg_attr[10] = np.mean(arr[:, 10]) # grad_t
            avg_attr[11] = np.mean(arr[:, 11]) # grad_p
            avg_attr[12] = np.min(arr[:, 12])  # m_t
            avg_attr[13] = np.min(arr[:, 13])  # m_p
            avg_attr[14] = np.mean(arr[:, 14]) # hm_visc
            avg_attr[15] = np.mean(arr[:, 15]) # hm_aniso_perm
            
            # TOF variations (t_cost is extracted in arr[:, 1] so it was originally the mean)
            # The A* algo itself appends t_cost at index 1.
            # So the min and max should also reference index 1.
            tof = arr[:, 1]
            avg_attr[16] = np.min(tof)   # min_tof
            avg_attr[17] = np.max(tof)   # max_tof

            edge_index_list.append([start_idx, goal_idx])
            edge_attr_list.append(avg_attr)
            if return_paths:
                path_coords_list.append(succ_paths)

    if len(edge_index_list) == 0:
        if return_paths:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0, 18), dtype=np.float32),
                [],
            )
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 18), dtype=np.float32)

    edge_index = np.array(edge_index_list, dtype=np.int64).T
    edge_attr = np.array(edge_attr_list, dtype=np.float32)

    if return_paths:
        return edge_index, edge_attr, path_coords_list
    return edge_index, edge_attr
