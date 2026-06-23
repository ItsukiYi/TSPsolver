r"""Improvement #4: Confidence-based Tour Fragment Freezing.

Finds edges where DIFUSCO and Christofides+2opt AGREE (both methods
independently produce the same edge) -> high-confidence -> FREEZE.

Only disputed edges are revised by DualOpt. This combines the strengths
of two independent solvers: if both choose the same edge, it's almost
certainly optimal.

Strategy:
  1. Run DIFUSCO: heatmap -> greedy merge -> DIFUSCO tour
  2. Run C+2opt: Christofides + 2-opt -> C+2opt tour
  3. Find edge intersection E_frozen = edges(DIFUSCO) INTERSECT edges(C+2opt)
  4. Run DualOpt reviser with frozen edges locked in place
"""

import numpy as np
import torch
from utils.functions import decomposition, revision


def get_edge_set(tour):
    """Convert tour (0-index vertex list, with or without return) to set of (min, max) edges."""
    edges = set()
    n = len(tour) if tour[-1] != tour[0] else len(tour) - 1
    for i in range(n):
        u = tour[i]
        v = tour[(i+1) % len(tour)] if tour[-1] == tour[0] else tour[(i+1) % n]
        if tour[-1] == tour[0] and (i == n-1):
            v = tour[0]
        edges.add((min(u, v), max(u, v)))
    return edges


def compute_frozen_mask(points, difusco_heatmap, c2opt_tour):
    """Find edges where DIFUSCO and C+2opt agree.

    Args:
        points: (n, 2) array
        difusco_heatmap: (n, n) array from DIFUSCO
        c2opt_tour: list of vertex indices (0-indexed, with return)

    Returns:
        frozen_mask: (n,) bool array — True for edges shared by both methods
        agreement_pct: float — percentage of edges agreed upon
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(points)

    # DIFUSCO tour via greedy merge
    # (simplified: use top-1 edge per node, handling ties)
    dif_edges = set()
    heatmap_sym = difusco_heatmap + difusco_heatmap.T
    # Greedy: for each node, pick highest-prob neighbor not yet taken
    used_nodes = set()
    np_pts = points.astype(np.float64)

    # Run simplified greedy merge from heatmap
    # Sort all possible edges by heatmap/distance
    dists = squareform(pdist(np_pts))
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            score = heatmap_sym[i, j] / (dists[i, j] + 1e-10)
            candidates.append((score, i, j))
    candidates.sort(reverse=True)

    degree = np.zeros(n, dtype=int)
    for score, i, j in candidates:
        if degree[i] < 2 and degree[j] < 2:
            dif_edges.add((i, j))
            degree[i] += 1
            degree[j] += 1
        if len(dif_edges) == n:
            break

    # C+2opt edges
    c2_edges = get_edge_set(c2opt_tour)

    # Intersection = frozen
    frozen_edges = dif_edges & c2_edges
    agreement_pct = len(frozen_edges) / n * 100

    # Convert to per-node-position mask (for tour order)
    # Use c2opt tour ordering as reference
    frozen_mask = np.zeros(n, dtype=bool)
    tour_no_return = c2opt_tour[:-1] if c2opt_tour[-1] == c2opt_tour[0] else c2opt_tour
    for i in range(n):
        u = tour_no_return[i]
        v = tour_no_return[(i+1) % n]
        edge = (min(u, v), max(u, v))
        if edge in frozen_edges:
            frozen_mask[i] = True

    return frozen_mask, agreement_pct


def freeze_guided_LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter,
                            frozen_mask):
    """Reviser with frozen edges: locked nodes stay in place.

    Only non-frozen nodes can be rearranged by the neural reviser.

    Args:
        seeds: (1, n, 2) tour coordinates in tour order
        cost_func: cost function
        reviser: neural reviser model
        revision_len: window size
        revision_iter: number of reviser passes
        frozen_mask: (n,) bool array — True = freeze edge at this position
    """
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    original_subtour = torch.arange(0, revision_len, dtype=torch.long, device=seeds.device)

    for iter_idx in range(revision_iter):
        shift_len = max(1, revision_len // revision_iter)
        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len=shift_len
        )

        orig = decomposed_seeds.clone()
        decomposed_seeds = revision(cost_func, reviser, decomposed_seeds, original_subtour)

        # Restore frozen nodes to original positions
        num_windows = decomposed_seeds.shape[0]
        for w in range(num_windows):
            for pos in range(revision_len):
                global_pos = (iter_idx * shift_len + w * revision_len + pos) % num_nodes
                if global_pos < num_nodes and frozen_mask[global_pos]:
                    decomposed_seeds[w, pos] = orig[w, pos]

        seeds = decomposed_seeds.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds, offset_seed], dim=1)

    return seeds
