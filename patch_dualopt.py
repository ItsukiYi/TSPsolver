"""Patch DualOpt-improved functions.py with edge-preserving reviser."""
import re

with open('DualOpt-improved/utils/functions.py', 'r', encoding='utf-8') as f:
    content = f.read()

start = content.find('def heatmap_guided_LCP_TSP')
end_marker = content.find('def second_step', start)

new_section = '''def heatmap_guided_LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter, heatmap, tour_perm=None, confidence_threshold=0.7):
    """Heatmap-guided LCP_TSP: preserves high-confidence edges during revision.

    Strategy: After each reviser pass, nodes belonging to high-confidence edges
    (DIFUSCO heatmap > threshold) are locked in place. Only uncertain nodes can
    be moved by the reviser, focusing optimization on genuinely difficult regions.
    """
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    original_subtour = torch.range(0, revision_len - 1, dtype=torch.long).cuda()

    edge_scores = compute_tour_edge_heatmap(seeds, heatmap, tour_permutation=tour_perm)
    confident_mask = (edge_scores > confidence_threshold)
    locked_positions = set(int(i) for i in range(num_nodes) if confident_mask[i].item())

    for iter_idx in range(revision_iter):
        shift_len = max(1, revision_len // revision_iter)
        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len=shift_len
        )

        orig = decomposed_seeds.clone()
        decomposed_seeds = revision(cost_func, reviser, decomposed_seeds, original_subtour)

        # Restore locked positions: confident nodes stay where they were
        num_windows = decomposed_seeds.shape[0]
        for w in range(num_windows):
            for pos in range(revision_len):
                global_pos = (iter_idx * shift_len + w * revision_len + pos) % num_nodes
                if global_pos in locked_positions:
                    decomposed_seeds[w, pos] = orig[w, pos]

        seeds = decomposed_seeds.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds, offset_seed], dim=1)

    return seeds

'''

new_content = content[:start] + new_section + content[start + end_marker:]
with open('DualOpt-improved/utils/functions.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Patched successfully: edge-preserving heatmap_guided_LCP_TSP')
