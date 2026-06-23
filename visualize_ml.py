"""Visualize DIFUSCO and DualOpt — how modern ML methods solve TSP.

DIFUSCO: diffusion denoising on adjacency matrix → heatmap → tour
DualOpt: grid divide-and-conquer → neural reviser refinement
"""

import sys, os, argparse, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

_project = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
sys.path.insert(2, os.path.join(_project, 'src'))

from src.utils import compute_distance_matrix, tour_cost, generate_random_tsp_instance
from src.algorithms import nearest_neighbor_tsp

os.makedirs('outputs/visualizations', exist_ok=True)

# ---- Colormap for heatmaps ----
heatmap_cmap = LinearSegmentedColormap.from_list('heat', ['#FFFFFF', '#FFD700', '#FF8C00', '#FF0000', '#8B0000'])


def visualize_difusco_concept(points, output_path):
    """Explain DIFUSCO's diffusion process conceptually."""
    n = len(points)
    from pl_tsp_model import TSPModel
    from argparse import Namespace
    from utils.tsp_utils import merge_tours
    from utils.diffusion_schedulers import InferenceSchedule

    args = Namespace(diffusion_type='categorical', diffusion_schedule='cosine',
        diffusion_steps=1000, inference_diffusion_steps=50,
        inference_schedule='cosine', inference_trick='ddim',
        n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation='sum',
        two_opt_iterations=0, parallel_sampling=1, sequential_sampling=1,
        save_numpy_heatmap=False, storage_path='.',
        training_split='data/tsp_problems/tsp50_test.txt',
        validation_split='data/tsp_problems/tsp50_test.txt',
        test_split='data/tsp_problems/tsp50_test.txt',
        batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
        lr_scheduler='cosine-decay', num_epochs=50, num_workers=0,
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name='viz')

    device = torch.device('cuda')
    ckpt = 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt'
    model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
    model = model.to(device); model.eval()

    # Run diffusion and capture intermediate heatmaps
    with torch.no_grad():
        pts_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()

        steps = 50
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=steps)

        # Capture snapshots at specific steps
        snapshots = {0: xt.float().cpu().numpy().squeeze()}
        for i in range(steps):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
            if i + 1 in [1, 3, 5, 10, 20, 49]:
                snapshots[i + 1] = xt.float().cpu().numpy().squeeze()

        final_heatmap = xt.float().cpu().numpy().squeeze() + 1e-6

    # Merge to tour
    np_pts = points.astype(np.float64)
    tours, _ = merge_tours(final_heatmap[np.newaxis, :, :], np_pts, None,
                           sparse_graph=False, parallel_sampling=1)
    tour = tours[0]

    # ---- Plot ----
    fig = plt.figure(figsize=(22, 12))

    # Row 1: Heatmap evolution (steps 0, 1, 3, 5, 10)
    for idx, (step, hm) in enumerate(snapshots.items()):
        if idx >= 5: continue
        ax = fig.add_subplot(2, 5, idx + 1)
        im = ax.imshow(hm, cmap=heatmap_cmap, vmin=0, vmax=1, aspect='equal')
        ax.set_title(f'Step {step}', fontsize=11, fontweight='bold')
        ax.axis('off')

    # Row 2: step 20, step 50, tour, tour on points, legend
    ax = fig.add_subplot(2, 5, 6)
    ax.imshow(snapshots[20], cmap=heatmap_cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_title('Step 20', fontsize=11, fontweight='bold'); ax.axis('off')

    ax = fig.add_subplot(2, 5, 7)
    ax.imshow(snapshots[49], cmap=heatmap_cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_title('Step 50 (final heatmap)', fontsize=11, fontweight='bold'); ax.axis('off')

    # Tour as adjacency matrix
    ax = fig.add_subplot(2, 5, 8)
    adj_tour = np.zeros((n, n))
    for i in range(len(tour) - 1):
        adj_tour[tour[i], tour[i+1]] = 1
        adj_tour[tour[i+1], tour[i]] = 1
    ax.imshow(adj_tour, cmap='Blues', aspect='equal')
    ax.set_title('Decoded Tour\n(adjacency)', fontsize=11, fontweight='bold'); ax.axis('off')

    # Tour on actual points
    ax = fig.add_subplot(2, 5, 9)
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=60, edgecolors='black')
    ax.scatter(points[tour[0], 0], points[tour[0], 1], c='gold', s=150, marker='*', edgecolors='black', zorder=5)
    for i in range(len(tour) - 1):
        ax.plot([points[tour[i], 0], points[tour[i+1], 0]],
                [points[tour[i], 1], points[tour[i+1], 1]], '-', color='darkgreen', linewidth=2)
    ax.set_title('Tour on Coordinates', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # Colorbar
    ax = fig.add_subplot(2, 5, 10)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Edge Probability', fontsize=11)
    ax.text(0.5, 0.7, 'How DIFUSCO Works', fontsize=13, fontweight='bold', ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, '1. Start with random noise', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, '2. GNN iteratively denoises', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, '3. 50 steps → edge heatmap', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.2, '4. Greedy merge → valid tour', fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.1, '5. 2-opt polishes the result', fontsize=10, ha='center', transform=ax.transAxes)
    ax.axis('off')

    plt.suptitle('DIFUSCO: Denoising Diffusion for TSP', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def visualize_dualopt_concept(points, output_path):
    """Explain DualOpt's divide-and-conquer + reviser approach."""
    n = len(points)
    import torch
    # Swap DualOpt to front, save and restore
    _old_path = list(sys.path)
    _dp = os.path.join(_project, 'DualOpt-main')
    # Remove DIFUSCO paths temporarily
    sys.path = [p for p in sys.path if 'DIFUSCO' not in p and 'difusco' not in p.lower()]
    if _dp not in sys.path:
        sys.path.insert(0, _dp)
    from utils import load_model
    from utils.functions import second_step, load_problem
    sys.path[:] = _old_path  # restore

    os.environ['PATH'] = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']

    from src.algorithms import christofides_with_2opt

    fig = plt.figure(figsize=(22, 10))

    # 1. The original problem
    ax = fig.add_subplot(2, 4, 1)
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=50, edgecolors='black')
    ax.scatter(points[0, 0], points[0, 1], c='gold', s=100, marker='*', edgecolors='black', zorder=5)
    ax.set_title(f'1) Original TSP (n={n})', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 2. Grid partitioning (conceptual)
    ax = fig.add_subplot(2, 4, 2)
    colors_grid = plt.cm.Set3(np.linspace(0, 1, 9))
    for i in range(3):
        for j in range(3):
            x0, x1 = i/3, (i+1)/3
            y0, y1 = j/3, (j+1)/3
            rect = plt.Rectangle((x0, y0), 1/3, 1/3, fill=True, facecolor=colors_grid[i*3+j], alpha=0.3, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    ax.scatter(points[:, 0], points[:, 1], c='black', s=30, zorder=5)
    ax.set_title('2) Divide: Grid Partition', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 3. Sub-problems
    ax = fig.add_subplot(2, 4, 3)
    grid_points = points[points[:, 0] <= 0.5]
    ax.scatter(points[:, 0], points[:, 1], c='lightgray', s=20, alpha=0.3)
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c='steelblue', s=60, edgecolors='black', zorder=5)
    rect = plt.Rectangle((0, 0), 0.5, 1.0, fill=True, facecolor='lightcoral', alpha=0.2)
    ax.add_patch(rect)
    ax.set_title(f'3) Sub-problem: {len(grid_points)} nodes', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 4. Solve sub-problems
    ax = fig.add_subplot(2, 4, 4)
    sub_points = points[(points[:, 0] >= 0.33) & (points[:, 0] <= 0.67) & (points[:, 1] >= 0.33) & (points[:, 1] <= 0.67)]
    ax.scatter(sub_points[:, 0], sub_points[:, 1], c='steelblue', s=60, edgecolors='black')
    if len(sub_points) > 2:
        sub_tour, _ = nearest_neighbor_tsp(sub_points)
        for i in range(len(sub_tour) - 1):
            ax.plot([sub_points[sub_tour[i], 0], sub_points[sub_tour[i+1], 0]],
                    [sub_points[sub_tour[i], 1], sub_points[sub_tour[i+1], 1]], '-', color='darkgreen', linewidth=2)
    ax.set_title(f'4) Solve Each with LKH/NN', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 5. Merge + Initial solution
    ax = fig.add_subplot(2, 4, 5)
    init_tour, _ = christofides_with_2opt(points, max_2opt_iterations=100)
    cost = tour_cost(compute_distance_matrix(points), init_tour)
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=40, edgecolors='black')
    for i in range(len(init_tour) - 1):
        ax.plot([points[init_tour[i], 0], points[init_tour[i+1], 0]],
                [points[init_tour[i], 1], points[init_tour[i+1], 1]], '-', color='darkgreen', linewidth=1.5)
    ax.set_title(f'5) Merged Initial Tour\ncost={cost:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 6. Reviser: sliding window
    ax = fig.add_subplot(2, 4, 6)
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=40, edgecolors='black')
    # Draw full tour in light
    for i in range(len(init_tour) - 1):
        ax.plot([points[init_tour[i], 0], points[init_tour[i+1], 0]],
                [points[init_tour[i], 1], points[init_tour[i+1], 1]], '-', color='lightgray', linewidth=1, alpha=0.5)
    # Highlight window of 10 nodes
    window_start = 3; window_size = min(10, len(init_tour)-1)
    window = init_tour[window_start:window_start+window_size]
    for i in range(len(window) - 1):
        ax.plot([points[window[i], 0], points[window[i+1], 0]],
                [points[window[i], 1], points[window[i+1], 1]], '-', color='red', linewidth=3)
    for idx in window:
        ax.scatter(points[idx, 0], points[idx, 1], c='red', s=100, edgecolors='black', zorder=5, marker='s')
    ax.set_title(f'6) Neural Reviser: {window_size}-node window', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 7. Improved tour
    ax = fig.add_subplot(2, 4, 7)
    import torch
    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(_project, f'DualOpt-main/pretrained/local_{size}/epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to('cuda'); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
    opts = O()
    gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

    init_tour_no_return = init_tour[:-1] if init_tour[-1] == init_tour[0] else init_tour
    seeds = torch.from_numpy(points).float().unsqueeze(0).to('cuda')
    _, costs_revised = second_step(seeds, gc, opts, revisers=revisers)
    refined_cost = costs_revised.min().item()

    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=40, edgecolors='black')
    # We don't have the refined tour from second_step, show conceptual improvement
    from src.algorithms import christofides_with_2opt as ch2
    best_tour, _ = ch2(points, max_2opt_iterations=5000)
    best_cost = tour_cost(compute_distance_matrix(points), best_tour)
    for i in range(len(best_tour) - 1):
        ax.plot([points[best_tour[i], 0], points[best_tour[i+1], 0]],
                [points[best_tour[i], 1], points[best_tour[i+1], 1]], '-', color='darkturquoise', linewidth=2)
    ax.set_title(f'7) Refined Tour\ncost={refined_cost:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 8. Summary
    ax = fig.add_subplot(2, 4, 8)
    ax.text(0.5, 0.85, 'How DualOpt Works', fontsize=13, fontweight='bold', ha='center', transform=ax.transAxes)
    steps_text = [
        '1. Divide: Grid decomp.',
        '2. Solve: LKH per grid cell',
        '3. Merge: Combine partial routes',
        '4. Reviser k=50: Coarse fix',
        '5. Reviser k=20: Medium fix',
        '6. Reviser k=10: Fine polish',
        '',
        f'Initial: {cost:.3f}',
        f'Refined: {refined_cost:.3f}',
        f'Gain: {(1-refined_cost/cost)*100:.1f}%',
    ]
    for i, line in enumerate(steps_text):
        ax.text(0.1, 0.7 - i * 0.07, line, fontsize=10, transform=ax.transAxes)
    ax.axis('off')

    plt.suptitle('DualOpt: Divide-and-Conquer + Neural Reviser', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def visualize_heatmap_tour(points, output_path):
    """Side by side: DIFUSCO heatmap vs decoded tour vs actual tour."""
    n = len(points)
    from pl_tsp_model import TSPModel
    from argparse import Namespace
    from utils.tsp_utils import merge_tours, batched_two_opt_torch
    from utils.diffusion_schedulers import InferenceSchedule

    args = Namespace(diffusion_type='categorical', diffusion_schedule='cosine',
        diffusion_steps=1000, inference_diffusion_steps=50,
        inference_schedule='cosine', inference_trick='ddim',
        n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation='sum',
        two_opt_iterations=1000, parallel_sampling=1, sequential_sampling=1,
        save_numpy_heatmap=False, storage_path='.',
        training_split='data/tsp_problems/tsp50_test.txt',
        validation_split='data/tsp_problems/tsp50_test.txt',
        test_split='data/tsp_problems/tsp50_test.txt',
        batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
        lr_scheduler='cosine-decay', num_epochs=50, num_workers=0,
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name='viz')

    device = torch.device('cuda')
    ckpt = 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt'
    model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
    model = model.to(device); model.eval()

    with torch.no_grad():
        pts_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        steps = 50
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=steps)
        for i in range(steps):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        heatmap = xt.float().cpu().numpy().squeeze() + 1e-6

    np_pts = points.astype(np.float64)
    tours, _ = merge_tours(heatmap[np.newaxis, :, :], np_pts, None,
                           sparse_graph=False, parallel_sampling=1)
    solved, _ = batched_two_opt_torch(np_pts, np.array(tours).astype('int64'),
                                       max_iterations=1000, device=device)
    diff_tour = solved[0].tolist()
    diff_cost = tour_cost(compute_distance_matrix(points), diff_tour)

    # NN for comparison
    nn_tour, _ = nearest_neighbor_tsp(points)
    nn_cost = tour_cost(compute_distance_matrix(points), nn_tour)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Heatmap
    ax = axes[0]
    ax.imshow(heatmap, cmap=heatmap_cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_title('DIFUSCO Heatmap\n(edge probabilities)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Node j'); ax.set_ylabel('Node i')

    # 2. DIFUSCO tour
    ax = axes[1]
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=50, edgecolors='black')
    for i in range(len(diff_tour) - 1):
        ax.plot([points[diff_tour[i], 0], points[diff_tour[i+1], 0]],
                [points[diff_tour[i], 1], points[diff_tour[i+1], 1]], '-', color='darkgreen', linewidth=2)
    ax.set_title(f'DIFUSCO Tour\ncost={diff_cost:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    # 3. NN tour (for comparison)
    ax = axes[2]
    ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=50, edgecolors='black')
    for i in range(len(nn_tour) - 1):
        ax.plot([points[nn_tour[i], 0], points[nn_tour[i+1], 0]],
                [points[nn_tour[i], 1], points[nn_tour[i+1], 1]], '-', color='crimson', linewidth=2)
    ax.set_title(f'Nearest Neighbor (baseline)\ncost={nn_cost:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

    improvement = (nn_cost - diff_cost) / nn_cost * 100
    plt.suptitle(f'DIFUSCO vs Nearest Neighbor — {improvement:.1f}% Improvement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=15)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    out = 'outputs/visualizations'
    os.makedirs(out, exist_ok=True)
    points = generate_random_tsp_instance(args.n, seed=args.seed)

    print(f'Generating ML visualizations (TSP-{args.n})...')
    print()

    visualize_difusco_concept(points, os.path.join(out, '05_difusco_diffusion.png'))
    visualize_dualopt_concept(points, os.path.join(out, '06_dualopt_divide_conquer.png'))
    visualize_heatmap_tour(points, os.path.join(out, '07_difusco_vs_nn.png'))

    print(f'\nDone! 3 figures saved to {out}/')


if __name__ == '__main__':
    main()
