"""Heatmap-Guided Reviser Module — Improvement #1 for DualOpt.

Uses DIFUSCO's edge probability heatmap to selectively skip revision
on windows where the initial tour already has high-confidence edges.
"""

import torch
from utils.functions import decomposition, revision


def compute_tour_edge_heatmap(seeds, heatmap, tour_permutation=None):
    """Extract per-edge heatmap confidence for the current tour.

    Args:
        seeds: (1, n, 2) tensor — tour coordinates in tour order
        heatmap: (n, n) numpy array — DIFUSCO edge probabilities [0,1]
        tour_permutation: (n,) list — maps tour position -> original node index

    Returns:
        edge_scores: (n,) tensor — heatmap score for each tour edge
    """
    n = seeds.shape[1]
    heatmap_t = torch.from_numpy(heatmap).float().to(seeds.device)

    if tour_permutation is not None:
        perm = torch.tensor(tour_permutation, dtype=torch.long, device=seeds.device)
        idx_from = perm
        idx_to = torch.cat([perm[1:], perm[:1]])
    else:
        idx_from = torch.arange(n, device=seeds.device)
        idx_to = torch.cat([torch.arange(1, n, device=seeds.device),
                            torch.zeros(1, dtype=torch.long, device=seeds.device)])

    edge_scores = heatmap_t[idx_from, idx_to]
    return edge_scores


def heatmap_guided_LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter,
                            heatmap, tour_perm=None, confidence_threshold=0.5):
    """Skip revision on windows where DIFUSCO heatmap shows high confidence.

    For each window, compute average edge confidence from the heatmap.
    Windows with avg confidence > threshold are left unchanged.
    Only uncertain windows are processed by the neural reviser.
    """
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    original_subtour = torch.arange(0, revision_len, dtype=torch.long, device=seeds.device)

    edge_scores = compute_tour_edge_heatmap(seeds, heatmap, tour_permutation=tour_perm)
    confidence_mask = (edge_scores > confidence_threshold).float()

    n_skipped = 0
    n_revised = 0

    for iter_idx in range(revision_iter):
        shift_len = max(1, revision_len // revision_iter)
        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len=shift_len
        )

        num_windows = decomposed_seeds.shape[0]
        for w in range(num_windows):
            win_start = (iter_idx * shift_len + w * revision_len) % num_nodes

            # Compute average confidence for edges in this window
            win_conf = 0.0
            win_count = 0
            for pos in range(revision_len - 1):
                gp = (win_start + pos) % num_nodes
                if gp < num_nodes:
                    win_conf += confidence_mask[gp].item()
                    win_count += 1
            avg_conf = win_conf / max(win_count, 1)

            if avg_conf > confidence_threshold:
                n_skipped += 1
                continue  # skip: window already good

            n_revised += 1
            single = decomposed_seeds[w:w+1]
            revised = revision(cost_func, reviser, single, original_subtour)
            decomposed_seeds[w] = revised[0]

        seeds = decomposed_seeds.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds, offset_seed], dim=1)

    return seeds
