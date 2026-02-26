import numpy as np
import heapq


def generate_geology_edges(
    perm_grid: np.ndarray,
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

    Args:
        perm_grid: 3D numpy array of shape [Z, X, Y] representing permeability
        porosity_grid: 3D numpy array of shape [Z, X, Y] representing porosity or faults.
        temp_grid: 3D numpy array of shape [Z, X, Y] representing initial temperature.
        press_grid: 3D numpy array of shape [Z, X, Y] representing initial pressure.
        is_injector: 1D boolean numpy array of shape [N] indicating if well is an injector.
        well_coords: 2D numpy array of shape [N, 3] representing the (X, Y, Z) indices
        k_neighbors: the number of closest injectors and extractors to route to.

    Returns:
        edge_index: integer array of shape [2, num_edges]
        edge_attr: float array of shape [num_edges, 14] containing:
                   (path length, harmonic poro / harmonic perm path cost (Time of flight),
                    min perm, max perm, harmonic mean perm,
                    min poro, max poro, harmonic mean poro,
                    delta_t, delta_p, grad_t, grad_p, min_temp, min_press)
    """
    num_wells = well_coords.shape[0]
    if num_wells < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 14), dtype=np.float32)

    # Convert coordinates to Z, X, Y format for direct grid indexing
    # well_coords are given as [X, Y, Z] based on our extraction code
    well_zxy = np.stack(
        [well_coords[:, 2], well_coords[:, 0], well_coords[:, 1]], axis=1
    )

    max_z, max_x, max_y = perm_grid.shape

    # Pre-compute valid mask to avoid moving through sealing faults
    min_perm_threshold = 1e-15
    valid_mask = (perm_grid > min_perm_threshold) & (porosity_grid > 0.0)

    # max_perm is used to scale distance heuristics
    max_perm_val = max(float(np.max(perm_grid)), 1e-12)
    # min acceptable porosity to avoid divide by zero during heuristic
    min_poro_val = (
        min(float(np.min(porosity_grid[valid_mask])), 1.0)
        if np.any(valid_mask)
        else 1e-4
    )

    # 3D 6-way neighborhood: (dz, dx, dy) and their corresponding physical distances
    # Assuming uniform grid blocks of 1.0 distance unit. If grid dimensions are non-uniform, this would need adjusting.
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
        # Admissible heuristic: under-estimates the true Time of Flight.
        # Shortest possible distance * smallest possible porosity / largest possible permeability
        return dist * (min_poro_val / max_perm_val)

    edge_index_list = []
    edge_attr_list = []
    path_coords_list = []

    target_goals = []
    # Pre-compute Euclidean distances to find k-nearest neighbors O(kN) routing
    for start_idx in range(num_wells):
        dists = np.sqrt(np.sum((well_coords[start_idx] - well_coords) ** 2, axis=1))

        # We want to connect to k closest injectors and k closest extractors
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

    # A* Algorithm Loop
    for start_idx, goal_idx in target_goals:
        start_pos = tuple(well_zxy[start_idx])
        goal_pos = tuple(well_zxy[goal_idx])

        # If start or goal are invalid, skip
        if not valid_mask[start_pos] or not valid_mask[goal_pos]:
            continue

        open_set = []
        heapq.heappush(open_set, (0.0, start_pos))

        cost_so_far = {start_pos: 0.0}
        came_from = {start_pos: None}

        # Keep track of path metrics
        path_length = {start_pos: 0}
        min_perm = {start_pos: perm_grid[start_pos]}
        max_perm = {start_pos: perm_grid[start_pos]}
        # True path harmonic mean: accumulate 1/k at each node, then N / sum(1/k)
        sum_inv_perm = {start_pos: 1.0 / max(float(perm_grid[start_pos]), 1e-30)}

        min_poro = {start_pos: porosity_grid[start_pos]}
        max_poro = {start_pos: porosity_grid[start_pos]}
        sum_inv_poro = {start_pos: 1.0 / max(float(porosity_grid[start_pos]), 1e-30)}

        # Safe-guard against -999 sentinel values in thermodynamic grids
        t_start_raw = temp_grid[start_pos]
        p_start_raw = press_grid[start_pos]
        t_start = t_start_raw if t_start_raw > -900 else 0.0
        p_start = p_start_raw if p_start_raw > -900 else 0.0

        min_temp = {start_pos: t_start}
        max_temp = {start_pos: t_start}
        min_press = {start_pos: p_start}
        max_press = {start_pos: p_start}

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

                # Physics Cost: Time of Flight = (Porosity / Permeability) * Distance
                step_cost = (hm_poro_edge / hm_perm_edge) * dist
                new_cost = cost_so_far[current_pos] + step_cost

                if n_pos not in cost_so_far or new_cost < cost_so_far[n_pos]:
                    cost_so_far[n_pos] = new_cost
                    priority = new_cost + _heuristic(n_pos, goal_pos)
                    heapq.heappush(open_set, (priority, n_pos))
                    came_from[n_pos] = current_pos

                    path_length[n_pos] = path_length[current_pos] + 1
                    min_perm[n_pos] = min(min_perm[current_pos], n_perm)
                    max_perm[n_pos] = max(max_perm[current_pos], n_perm)
                    sum_inv_perm[n_pos] = sum_inv_perm[current_pos] + 1.0 / max(
                        float(n_perm), 1e-30
                    )

                    min_poro[n_pos] = min(min_poro[current_pos], n_poro)
                    max_poro[n_pos] = max(max_poro[current_pos], n_poro)
                    sum_inv_poro[n_pos] = sum_inv_poro[current_pos] + 1.0 / max(
                        float(n_poro), 1e-30
                    )

                    # Clamp sentinel -999 values before tracking thermodynamic extremes
                    n_temp = temp_grid[nz, nx, ny]
                    n_press = press_grid[nz, nx, ny]
                    n_temp = n_temp if n_temp > -900 else min_temp[current_pos]
                    n_press = n_press if n_press > -900 else min_press[current_pos]

                    min_temp[n_pos] = min(min_temp[current_pos], n_temp)
                    max_temp[n_pos] = max(max_temp[current_pos], n_temp)
                    min_press[n_pos] = min(min_press[current_pos], n_press)
                    max_press[n_pos] = max(max_press[current_pos], n_press)

        if found_path:
            plen = path_length[goal_pos]
            # Number of nodes on the path = edges + 1
            n_nodes = plen + 1

            # True path harmonic mean for series flow: N / Σ(1/kᵢ)
            path_hm_perm = n_nodes / sum_inv_perm[goal_pos]
            m_perm = min_perm[goal_pos]
            mx_perm = max_perm[goal_pos]

            path_hm_poro = n_nodes / sum_inv_poro[goal_pos]
            m_poro = min_poro[goal_pos]
            mx_poro = max_poro[goal_pos]

            m_t = min_temp[goal_pos]
            m_p = min_press[goal_pos]

            # Endpoint delta: pressure/temperature head driving flow (Darcy's law proxy)
            t_goal_raw = temp_grid[goal_pos]
            p_goal_raw = press_grid[goal_pos]
            t_goal = t_goal_raw if t_goal_raw > -900 else t_start
            p_goal = p_goal_raw if p_goal_raw > -900 else p_start

            delta_t = t_goal - t_start
            delta_p = p_goal - p_start

            grad_t = delta_t / max(plen, 1)
            grad_p = delta_p / max(plen, 1)

            t_cost = cost_so_far[goal_pos]

            edge_index_list.append([start_idx, goal_idx])
            edge_attr_list.append(
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
                    m_p,  # 14 dimensional edge feature array
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
                path_coords_list.append(out_p)

    if len(edge_index_list) == 0:
        if return_paths:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0, 14), dtype=np.float32),
                [],
            )
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 14), dtype=np.float32)

    edge_index = np.array(edge_index_list, dtype=np.int64).T
    edge_attr = np.array(edge_attr_list, dtype=np.float32)

    if return_paths:
        return edge_index, edge_attr, path_coords_list
    return edge_index, edge_attr
