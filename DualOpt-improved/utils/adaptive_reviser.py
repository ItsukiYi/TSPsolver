"""Improvement #3: Adaptive Window Sizing for DualOpt Reviser.

Uses a 2-opt diagnostic pass to identify "unstable" tour segments
(edges that 2-opt can improve) and dynamically allocates more reviser
iterations to unstable regions while skipping already-optimal segments.

Strategy:
  1. Run lightweight 2-opt (10 iterations) on current tour
  2. Track which edges were changed → "unstable" mask
  3. Allocate reviser iterations proportional to instability density
  4. Stable windows get fewer iterations; unstable windows get more
"""

import torch
import numpy as np
from utils.functions import decomposition, revision


def run_2opt_diagnostic(points, tour, max_iterations=10, device='cuda'):
    """Run lightweight 2-opt and return which edges were changed.

    Args:
        points: (n, 2) numpy array
        tour: (n,) numpy array of 0-indexed vertices (no return-to-start)
        max_iterations: max 2-opt iterations (keep low for diagnostic)

    Returns:
        changed_mask: (n,) bool array — True if edge (tour[i], tour[i+1]) was changed
    """
    n = len(tour)
    # Track original edges
    orig_edges = set()
    for i in range(n):
        u, v = tour[i], tour[(i+1) % n]
        orig_edges.add((min(u, v), max(u, v)))

    # Run 2-opt on GPU
    pts_t = torch.from_numpy(points).float().to(device)
    tour_t = torch.from_numpy(tour.copy()).long().unsqueeze(0).to(device)

    with torch.inference_mode():
        for _ in range(max_iterations):
            pi = pts_t[tour_t[0, :-1]].reshape(1, -1, 1, 2)
            pj = pts_t[tour_t[0, :-1]].reshape(1, 1, -1, 2)
            pi1 = pts_t[tour_t[0, 1:]].reshape(1, -1, 1, 2)
            pj1 = pts_t[tour_t[0, 1:]].reshape(1, 1, -1, 2)

            A_ij = torch.sqrt(((pi - pj)**2).sum(-1))
            A_i1j1 = torch.sqrt(((pi1 - pj1)**2).sum(-1))
            A_ii1 = torch.sqrt(((pi - pi1)**2).sum(-1))
            A_jj1 = torch.sqrt(((pj - pj1)**2).sum(-1))

            change = A_ij + A_i1j1 - A_ii1 - A_jj1
            valid_change = torch.triu(change, diagonal=2)
            min_val = torch.min(valid_change)

            if min_val >= -1e-10:
                break

            flat_idx = torch.argmin(valid_change.reshape(1, -1), dim=-1)
            min_i = flat_idx // n
            min_j = flat_idx % n
            tour_t[0, min_i[0]+1:min_j[0]+1] = torch.flip(
                tour_t[0, min_i[0]+1:min_j[0]+1], dims=(0,))

    final_tour = tour_t[0].cpu().numpy()

    # Compare: which edges changed?
    changed = np.zeros(n, dtype=bool)
    for i in range(n):
        u_new = final_tour[i]
        v_new = final_tour[(i+1) % n]
        edge_new = (min(u_new, v_new), max(u_new, v_new))
        if edge_new not in orig_edges:
            changed[i] = True

    return changed


def adaptive_window_LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter,
                             points_np, tour_np, instability_threshold=0.2):
    """Adaptive window sizing: unstable segments get more reviser iterations.

    Args:
        seeds: (1, n, 2) tour coordinates in tour order
        cost_func: cost function
        reviser: neural reviser model
        revision_len: base window size (e.g., 20)
        revision_iter: base iteration count (e.g., 10)
        points_np: (n, 2) original coordinates (for 2-opt diagnostic)
        tour_np: (n,) current tour permutation (0-indexed, no return)
        instability_threshold: windows above this fraction of unstable edges get boosted

    Returns:
        seeds: refined tour
    """
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    original_subtour = torch.arange(0, revision_len, dtype=torch.long, device=seeds.device)

    # ---- Diagnostic: identify unstable edges via 2-opt ----
    changed_mask = run_2opt_diagnostic(points_np, tour_np, max_iterations=10)

    # Sliding window instability scores
    n_windows_total = num_nodes  # approx: one per possible start position
    window_scores = np.zeros(n_windows_total)
    for start in range(min(n_windows_total, num_nodes)):
        end = start + revision_len
        if end <= num_nodes:
            unstable_count = changed_mask[start:end-1].sum()
        else:
            # wrap-around window
            unstable_count = (changed_mask[start:].sum() +
                            changed_mask[:end - num_nodes - 1].sum())
        window_scores[start] = unstable_count / max(revision_len - 1, 1)

    # Dynamic iteration budget per window position
    base_shift = max(1, revision_len // revision_iter)
    total_budget = revision_iter * (num_nodes // revision_len + 1)
    allocated = 0

    # Sort windows by instability (descending) and allocate budget
    window_order = np.argsort(-window_scores)  # most unstable first
    window_iters = np.zeros(n_windows_total, dtype=int)
    budget_remaining = total_budget
    for w in window_order:
        if budget_remaining <= 0:
            break
        score = window_scores[w]
        if score > instability_threshold:
            alloc = min(2, budget_remaining)  # unstable: up to 2 extra iters
        elif score > 0:
            alloc = min(1, budget_remaining)  # some instability: 1 iter
        else:
            alloc = 0  # stable: skip
        window_iters[w] = alloc
        budget_remaining -= alloc
        allocated += alloc

    # ---- Execute reviser with adaptive iteration counts ----
    for iter_idx in range(revision_iter):
        shift_len = max(1, revision_len // revision_iter)
        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len=shift_len
        )

        num_windows = decomposed_seeds.shape[0]
        for w in range(num_windows):
            win_start = (iter_idx * shift_len + w * revision_len) % num_nodes
            # Only revise if this window has budget remaining
            if iter_idx < window_iters[min(win_start, n_windows_total - 1)]:
                # Extra revision for unstable windows
                single = decomposed_seeds[w:w+1]
                revised = revision(cost_func, reviser, single, original_subtour)
                decomposed_seeds[w] = revised[0]
            # else: window already revised enough

        seeds = decomposed_seeds.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds, offset_seed], dim=1)

    return seeds
