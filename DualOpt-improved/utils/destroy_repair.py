r"""Improvement #5: Destroy-and-Repair with DIFUSCO Heatmap Targeting.

Inspired by DRHG (Li et al., AAAI 2025) and ruin-and-recreate heuristics.

Algorithm:
  1. DIFUSCO heatmap -> per-edge confidence scores along current tour
  2. DESTROY: Remove K edges with lowest confidence, breaking tour into K segments
  3. REPAIR: Reconnect segments using nearest-neighbor greedy + 2-opt polish
  4. REPEAT: Iterate destroy-repair cycle for M epochs

Key innovation vs Improvement #1:
  - #1: PASSIVELY skip revision on confident windows -> no effect
  - #5: ACTIVELY destroy uncertain edges and rebuild -> forces exploration
"""

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


def compute_edge_confidence(heatmap, tour):
    """Compute per-edge confidence from DIFUSCO heatmap along tour.

    Args:
        heatmap: (n, n) array - DIFUSCO edge probabilities
        tour: list of vertex indices (with or without return-to-start)

    Returns:
        confidences: (m,) array where m = len(tour)-1 (edges)
    """
    n = len(tour) if tour[-1] != tour[0] else len(tour) - 1
    confs = np.zeros(n)
    for i in range(n):
        u = tour[i]
        v = tour[(i+1) % len(tour)]
        confs[i] = heatmap[u, v]
    return confs


def destroy(tour, confidences, K):
    """Remove K lowest-confidence edges, returning connected segments.

    Edge at position i connects tour[i] → tour[(i+1) % n].
    Destroying edge i means the tour breaks between i and i+1.

    Args:
        tour: list of vertex indices (NO return-to-start)
        confidences: (n,) array of per-edge confidence
        K: number of edges to destroy

    Returns:
        segments: list of lists, each a contiguous path (guaranteed to
                  contain all nodes exactly once)
        destroyed_indices: sorted indices of destroyed edges
    """
    n = len(tour)
    edge_indices = sorted(np.argsort(confidences)[:K])
    destroyed = set(edge_indices)

    # Walk tour, splitting at destroyed edges
    segments = []
    current = [tour[0]]

    for i in range(n - 1):
        if i in destroyed:
            # Edge i (tour[i]→tour[i+1]) destroyed: split here
            segments.append(current)
            current = [tour[i + 1]]
        else:
            current.append(tour[i + 1])

    # Handle edge n-1 (tour[n-1]→tour[0]): merge last with first if intact
    if (n - 1) in destroyed:
        segments.append(current)
    else:
        # Last edge intact: last segment connects to first
        if len(segments) > 0:
            segments[0] = current + segments[0]
        else:
            segments.append(current)

    # Sanity check: all nodes present exactly once
    all_nodes = [v for s in segments for v in s]
    assert len(all_nodes) == n, f"Destroy bug: {len(all_nodes)} nodes, expected {n}"
    assert set(all_nodes) == set(tour), "Destroy bug: missing or duplicate nodes"

    return segments, edge_indices


def repair_greedy(segments, points):
    """Reconnect segments using nearest-neighbor greedy.

    Args:
        segments: list of lists (each a contiguous path)
        points: (n, 2) coordinate array

    Returns:
        tour: list of vertex indices (no return)
        cost: float
    """
    if len(segments) <= 1:
        tour = segments[0] if segments else []
        return tour, 0.0

    dists = squareform(pdist(points))
    K = len(segments)

    # Each segment has: start=seg[0], end=seg[-1]
    starts = [s[0] for s in segments]
    ends = [s[-1] for s in segments]

    # Greedy: start from segment 0, always connect to nearest unvisited segment start
    visited = [False] * K
    visited[0] = True
    order = [0]
    current_end = ends[0]

    for _ in range(K - 1):
        best_dist = float('inf')
        best_idx = -1
        best_use_start = True
        for i in range(K):
            if visited[i]:
                continue
            # Connect current_end -> starts[i] (then traverse segment i forward)
            d_forward = dists[current_end, starts[i]]
            # Connect current_end -> ends[i] (then traverse segment i backward)
            d_backward = dists[current_end, ends[i]]
            if d_forward < best_dist:
                best_dist = d_forward
                best_idx = i
                best_use_start = True
            if d_backward < best_dist:
                best_dist = d_backward
                best_idx = i
                best_use_start = False

        visited[best_idx] = True
        order.append(best_idx)
        if best_use_start:
            current_end = ends[best_idx]
        else:
            # Need to reverse this segment
            segments[best_idx] = list(reversed(segments[best_idx]))
            current_end = starts[best_idx]

    # Build tour from ordered segments
    tour = []
    for idx in order:
        tour.extend(segments[idx])

    # Compute cost
    cost = 0.0
    for i in range(len(tour)):
        cost += dists[tour[i], tour[(i+1) % len(tour)]]

    return tour, cost


def repair_2opt(tour, points, max_iter=1000):
    """Polish repaired tour with 2-opt."""
    n = len(tour)
    pts_t = torch.from_numpy(points).float().to('cuda')
    tour_t = torch.tensor(tour, dtype=torch.long, device='cuda').unsqueeze(0)

    with torch.inference_mode():
        for _ in range(max_iter):
            pi = pts_t[tour_t[0, :-1]].reshape(1, -1, 1, 2)
            pj = pts_t[tour_t[0, :-1]].reshape(1, 1, -1, 2)
            pi1 = pts_t[tour_t[0, 1:]].reshape(1, -1, 1, 2)
            pj1 = pts_t[tour_t[0, 1:]].reshape(1, 1, -1, 2)

            change = ((pi-pj)**2).sum(-1).sqrt() + ((pi1-pj1)**2).sum(-1).sqrt() - \
                     ((pi-pi1)**2).sum(-1).sqrt() - ((pj-pj1)**2).sum(-1).sqrt()
            valid = torch.triu(change, diagonal=2)
            min_val = torch.min(valid)

            if min_val >= -1e-10:
                break

            flat = torch.argmin(valid.reshape(1, -1), dim=-1)
            mi, mj = flat // n, flat % n
            tour_t[0, mi[0]+1:mj[0]+1] = torch.flip(tour_t[0, mi[0]+1:mj[0]+1], dims=(0,))

    return tour_t[0].cpu().tolist()


def destroy_repair_cycle(points, initial_tour, heatmap, K_values=None, cycles=3,
                          dualopt_revisers=None, dualopt_opts=None, verbose=False):
    """Full destroy-and-repair cycle with multi-K search and optional DualOpt polish.

    Tries multiple destroy sizes (K_values), picks the best repair for each cycle.

    Args:
        points: (n, 2) coordinates
        initial_tour: list of vertex indices (no return)
        heatmap: (n, n) DIFUSCO edge probabilities
        K_values: list of K (destroy count) to try per cycle, e.g. [3, 5, 8]
        cycles: number of destroy-repair iterations
        dualopt_revisers: optional DualOpt reviser models for final polish
        dualopt_opts: options for DualOpt reviser
        verbose: print progress

    Returns:
        best_tour: list of vertex indices (no return)
        best_cost: float
        history: list of costs per cycle
    """
    if K_values is None:
        K_values = [3, 5, 7]

    dists = squareform(pdist(points))
    tour = list(initial_tour)
    if tour[-1] == tour[0]:
        tour = tour[:-1]
    n = len(tour)

    best_tour = list(tour)
    best_cost = sum(dists[tour[i], tour[(i+1) % n]] for i in range(n))
    history = [best_cost]

    for cycle in range(cycles):
        confs = compute_edge_confidence(heatmap, tour)

        # Try multiple K values, pick the best repair
        best_cycle_cost = float('inf')
        best_cycle_tour = None

        for K in K_values:
            if K >= n // 3:  # Don't destroy more than 1/3 of edges
                continue
            segments, destroyed = destroy(tour, confs, K)
            repaired, _ = repair_greedy(segments, points)
            repaired = repair_2opt(repaired, points, max_iter=200)

            cost = sum(dists[repaired[i], repaired[(i+1) % n]] for i in range(n))
            if cost < best_cycle_cost:
                best_cycle_cost = cost
                best_cycle_tour = repaired

        if best_cycle_tour is None:
            continue

        if verbose:
            impr = (history[-1] - best_cycle_cost) / history[-1] * 100
            print(f'  Cycle {cycle+1}: {history[-1]:.4f} -> {best_cycle_cost:.4f} '
                  f'({impr:+.2f}%)')

        if best_cycle_cost < best_cost - 1e-8:
            best_cost = best_cycle_cost
            best_tour = list(best_cycle_tour)

        tour = list(best_cycle_tour)
        history.append(best_cycle_cost)

    # Optional: DualOpt reviser polish on best tour
    if dualopt_revisers is not None and dualopt_opts is not None:
        from utils.functions import second_step, load_problem
        gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
        seeds = torch.from_numpy(points[best_tour]).float().unsqueeze(0)
        if torch.cuda.is_available():
            seeds = seeds.to('cuda')
        _, cost_polished = second_step(seeds, gc, dualopt_opts, dualopt_revisers)
        cost_p = cost_polished.min().item()
        if cost_p < best_cost:
            if verbose:
                print(f'  DualOpt polish: {best_cost:.4f} -> {cost_p:.4f} '
                      f'({(best_cost-cost_p)/best_cost*100:+.2f}%)')
            best_cost = cost_p
            # Tour order may have changed; we return the cost improvement

    return best_tour, best_cost, history
