"""TSP Algorithm Visualization — step-by-step walkthrough for each method.

Generates annotated plots showing how each algorithm constructs/refines a TSP tour.
Uses a small instance (n=15) so the process is visually clear.

Usage:
    python visualize.py                    # generate all figures
    python visualize.py --animate          # generate animation frames (needs more time)
"""

import sys, os, argparse, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx

_project = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_project, 'src'))
from src.utils import compute_distance_matrix, tour_cost, generate_random_tsp_instance
from src.algorithms import nearest_neighbor_tsp, christofides_tsp, two_opt_tsp, christofides_with_2opt

# ---- Styling ----
COLORS = {
    'unvisited': '#87CEEB',    # light blue
    'visited': '#4169E1',      # royal blue
    'current': '#FF4444',      # red
    'path': '#2E8B57',         # sea green
    'depot': '#FFD700',        # gold
    'candidate': '#FFA500',    # orange
    'mst': '#8B4513',          # saddle brown
    'matching': '#FF69B4',     # hot pink
    'improved': '#00CED1',     # dark turquoise
    'removed': '#FF0000',      # red
}
FIGSIZE = (10, 8)


def plot_base(points, ax, title=''):
    """Draw base layer: points with labels."""
    n = len(points)
    ax.scatter(points[0, 0], points[0, 1], c=COLORS['depot'], s=200,
              marker='*', edgecolors='black', linewidth=1.5, zorder=5, label='Depot (start)')
    for i in range(1, n):
        ax.scatter(points[i, 0], points[i, 1], c=COLORS['unvisited'], s=80,
                  edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(str(i), (points[i, 0], points[i, 1]), fontsize=8,
                   ha='center', va='center', fontweight='bold')
    ax.annotate('0', (points[0, 0], points[0, 1]), fontsize=9,
               ha='center', va='center', fontweight='bold', color='white')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal'); ax.axis('off')


def draw_tour(ax, points, tour, color=COLORS['path'], alpha=0.7, linewidth=2):
    """Draw a complete tour."""
    for i in range(len(tour) - 1):
        ax.plot([points[tour[i], 0], points[tour[i+1], 0]],
                [points[tour[i], 1], points[tour[i+1], 1]],
                '-', color=color, alpha=alpha, linewidth=linewidth)


# ================================================================
# Nearest Neighbor — Step by Step
# ================================================================
def visualize_nn(points, output_path):
    """Show NN construction: at each step, highlight current node and next choice."""
    n = len(points)
    dist_mat = compute_distance_matrix(points)
    tour = [0]; visited = {0}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Step 0: initial state
    ax = axes[0]
    plot_base(points, ax, 'Step 0: Start at Depot (0)')
    ax.scatter(points[0, 0], points[0, 1], c=COLORS['depot'], s=200, marker='*', edgecolors='black')

    key_steps = [0, 1, n//4, n//2, 3*n//4, n-2, n-1]
    step_idx = 1

    for step in range(n - 1):
        current = tour[-1]
        unvisited = [i for i in range(n) if i not in visited]
        dists = [(dist_mat[current, j], j) for j in unvisited]
        dists.sort()
        nearest = dists[0][1]
        tour.append(nearest)
        visited.add(nearest)

        if step + 1 in key_steps and step_idx < 8:
            ax = axes[step_idx]
            plot_base(points, ax, f'Step {step+1}: Visit node {nearest}')
            for i in range(len(tour) - 1):
                ax.plot([points[tour[i], 0], points[tour[i+1], 0]],
                        [points[tour[i], 1], points[tour[i+1], 1]],
                        '-', color=COLORS['path'], alpha=0.7, linewidth=2)
            ax.scatter(points[current, 0], points[current, 1], c=COLORS['current'],
                      s=120, edgecolors='black', zorder=4)
            # Show candidate edges (dashed)
            for d, j in dists[:3]:
                ax.plot([points[current, 0], points[j, 0]],
                        [points[current, 1], points[j, 1]],
                        '--', color=COLORS['candidate'], alpha=0.3, linewidth=1)
            step_idx += 1

    tour.append(0)
    # Final step: complete tour
    ax = axes[7]
    plot_base(points, ax, f'Final: Complete Tour (cost={tour_cost(dist_mat, tour):.3f})')
    draw_tour(ax, points, tour)

    plt.suptitle('Nearest Neighbor Algorithm — Step by Step', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ================================================================
# Christofides — Key Stages
# ================================================================
def visualize_christofides(points, output_path):
    """Show Christofides stages: MST, odd vertices, matching, Eulerian, shortcut."""
    n = len(points)
    dist_mat = compute_distance_matrix(points)
    from scipy.sparse.csgraph import minimum_spanning_tree

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Stage 1: Complete graph → MST
    ax = axes[0]
    plot_base(points, ax, '1) Build Minimum Spanning Tree')
    mst_sparse = minimum_spanning_tree(dist_mat)
    mst_graph = nx.from_scipy_sparse_array(mst_sparse)
    for u, v in mst_graph.edges():
        ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]],
                '-', color=COLORS['mst'], linewidth=2.5, alpha=0.8)

    # Stage 2: Find odd-degree vertices
    ax = axes[1]
    plot_base(points, ax, '2) Identify Odd-Degree Vertices')
    for u, v in mst_graph.edges():
        ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]],
                '-', color=COLORS['mst'], linewidth=1.5, alpha=0.4)
    odd_vertices = [v for v in range(n) if mst_graph.degree(v) % 2 == 1]
    for v in odd_vertices:
        ax.scatter(points[v, 0], points[v, 1], c=COLORS['candidate'], s=150,
                  marker='o', edgecolors='red', linewidth=2, zorder=5)
    ax.set_title(f'2) Odd-Degree Vertices ({len(odd_vertices)} found)', fontsize=12, fontweight='bold')

    # Stage 3: Minimum-Weight Perfect Matching
    ax = axes[2]
    plot_base(points, ax, '3) Add Perfect Matching (MST + MWPM)')
    G_odd = nx.Graph()
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            if i < j:
                G_odd.add_edge(u, v, weight=-dist_mat[u, v])
    matching = nx.max_weight_matching(G_odd, maxcardinality=True)
    # Draw MST
    for u, v in mst_graph.edges():
        ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]],
                '-', color=COLORS['mst'], linewidth=1.5, alpha=0.3)
    # Draw matching
    for u, v in matching:
        ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]],
                '--', color=COLORS['matching'], linewidth=3, alpha=0.9)
    for v in odd_vertices:
        ax.scatter(points[v, 0], points[v, 1], c=COLORS['candidate'], s=100, zorder=4)

    # Stage 4: Eulerian Circuit
    ax = axes[3]
    eulerian_graph = nx.MultiGraph()
    for u, v in mst_graph.edges():
        eulerian_graph.add_edge(u, v)
    for u, v in matching:
        eulerian_graph.add_edge(u, v)
    plot_base(points, ax, f'4) Eulerian Circuit (all degrees even)')
    for u, v in eulerian_graph.edges():
        ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]],
                '-', color='gray', linewidth=0.8, alpha=0.5)

    # Stage 5: Shortcutting
    ax = axes[4]
    plot_base(points, ax, '5) Shortcut to Hamiltonian Cycle')
    eulerian_edges = list(nx.eulerian_circuit(eulerian_graph))
    visited = [False] * n; tour = []
    for u, v in eulerian_edges:
        if not visited[u]: tour.append(u); visited[u] = True
        if not visited[v]: tour.append(v); visited[v] = True
    tour.append(tour[0])
    draw_tour(ax, points, tour, color=COLORS['path'])

    # Stage 6: Final after 2-opt
    ax = axes[5]
    refined, _ = two_opt_tsp(points, tour, max_iterations=1000)
    cost = tour_cost(dist_mat, refined)
    plot_base(points, ax, f'6) After 2-opt Refinement (cost={cost:.3f})')
    draw_tour(ax, points, refined, color=COLORS['improved'])

    plt.suptitle('Christofides Algorithm — Key Stages', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ================================================================
# 2-opt — Before/After
# ================================================================
def visualize_2opt(points, output_path):
    """Show 2-opt improvement: initial tour vs refined, highlight a swap."""
    n = len(points)
    dist_mat = compute_distance_matrix(points)
    from src.algorithms import nearest_neighbor_tsp

    initial_tour, _ = nearest_neighbor_tsp(points)
    initial_cost = tour_cost(dist_mat, initial_tour)
    refined_tour, meta = two_opt_tsp(points, initial_tour, max_iterations=1000, verbose=False)
    refined_cost = tour_cost(dist_mat, refined_tour)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Before
    ax = axes[0]
    plot_base(points, ax, f'Before: Nearest Neighbor (cost={initial_cost:.3f})')
    draw_tour(ax, points, initial_tour)

    # During: illustrate a 2-opt swap
    ax = axes[1]
    plot_base(points, ax, '2-opt Move: Remove 2 edges, Reconnect')
    # Find two crossing edges
    mid = n // 2
    i, j = 2, mid
    # Current edges
    ax.plot([points[initial_tour[i], 0], points[initial_tour[i+1], 0]],
            [points[initial_tour[i], 1], points[initial_tour[i+1], 1]],
            '-', color=COLORS['removed'], linewidth=3, alpha=0.8, label='Remove')
    ax.plot([points[initial_tour[j], 0], points[initial_tour[(j+1)%n], 0]],
            [points[initial_tour[j], 1], points[initial_tour[(j+1)%n], 1]],
            '-', color=COLORS['removed'], linewidth=3, alpha=0.8)
    # New edges
    ax.plot([points[initial_tour[i], 0], points[initial_tour[j], 0]],
            [points[initial_tour[i], 1], points[initial_tour[j], 1]],
            '--', color=COLORS['improved'], linewidth=3, alpha=0.9, label='New')
    ax.plot([points[initial_tour[i+1], 0], points[initial_tour[(j+1)%n], 0]],
            [points[initial_tour[i+1], 1], points[initial_tour[(j+1)%n], 1]],
            '--', color=COLORS['improved'], linewidth=3, alpha=0.9)
    ax.legend(fontsize=10)

    # After
    ax = axes[2]
    plot_base(points, ax, f'After: 2-opt Refined (cost={refined_cost:.3f})')
    draw_tour(ax, points, refined_tour, color=COLORS['improved'])
    improvement = (initial_cost - refined_cost) / initial_cost * 100
    ax.text(0.5, -0.02, f'Improvement: {improvement:.1f}% ({meta["iterations"]} iterations)',
            transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

    plt.suptitle('2-opt Local Search — Before and After', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ================================================================
# Final Comparison — All Methods Side by Side
# ================================================================
def visualize_comparison(points, output_path):
    """Side-by-side comparison of all algorithms on the same instance."""
    n = len(points)
    dist_mat = compute_distance_matrix(points)

    methods = {
        'Nearest Neighbor': lambda: nearest_neighbor_tsp(points)[0],
        'Christofides': lambda: christofides_tsp(points)[0],
        'Christofides + 2-opt': lambda: christofides_with_2opt(points)[0],
    }

    fig, axes = plt.subplots(1, len(methods), figsize=(18, 6))
    for ax, (name, fn) in zip(axes, methods.items()):
        tour = fn()
        cost = tour_cost(dist_mat, tour)
        plot_base(points, ax, f'{name}\nCost: {cost:.3f}')
        draw_tour(ax, points, tour)
        # Show self-crossings in red (bad edges)
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour) - 1):
                if j == i + 1: continue
                if _segments_intersect(points, tour[i], tour[i+1], tour[j], tour[(j+1)%n]):
                    ax.scatter([points[tour[i], 0], points[tour[i+1], 0],
                               points[tour[j], 0], points[tour[(j+1)%n], 0]],
                              [points[tour[i], 1], points[tour[i+1], 1],
                               points[tour[j], 1], points[tour[(j+1)%n], 1]],
                              c='red', s=60, zorder=6, marker='x', linewidth=2)

    plt.suptitle('Algorithm Comparison — Same Instance', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


def _segments_intersect(points, a, b, c, d):
    """Check if segments (a,b) and (c,d) intersect (exclude shared endpoints)."""
    p1, p2 = points[a], points[b]
    p3, p4 = points[c], points[d]
    def _ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


# ================================================================
# How to Read a TSP Tour
# ================================================================
def visualize_how_to_read(points, output_path):
    """Explainer: what is a TSP instance and how to read a tour."""
    n = len(points)
    dist_mat = compute_distance_matrix(points)
    tour, _ = nearest_neighbor_tsp(points)
    cost = tour_cost(dist_mat, tour)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. The problem
    ax = axes[0]
    plot_base(points, ax, 'The Problem: Visit Every Point Once')
    ax.text(0.5, -0.05, f'{n} delivery locations in a city. Start & end at depot (star).',
            transform=ax.transAxes, ha='center', fontsize=11)

    # 2. A solution
    ax = axes[1]
    plot_base(points, ax, 'A Solution: A Hamiltonian Cycle')
    draw_tour(ax, points, tour)
    arrows_added = 0
    for i in range(min(len(tour) - 1, 4)):
        u, v = tour[i], tour[i+1]
        ax.annotate('', xy=(points[v, 0], points[v, 1]),
                   xytext=(points[u, 0], points[u, 1]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # 3. The objective
    ax = axes[2]
    plot_base(points, ax, 'The Goal: Minimize Total Distance')
    draw_tour(ax, points, tour)
    # Annotate a few edge lengths
    for i in range(min(3, len(tour) - 1)):
        u, v = tour[i], tour[i+1]
        d = dist_mat[u, v]
        mid = (points[u] + points[v]) / 2
        ax.annotate(f'{d:.2f}', (mid[0], mid[1]), fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    ax.text(0.5, -0.05, f'Total route length: {cost:.3f} units',
            transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('TSP for Beginners — What Are We Solving?', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='TSP Algorithm Visualization')
    parser.add_argument('--n', type=int, default=15, help='Number of nodes')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    points = generate_random_tsp_instance(args.n, seed=args.seed)

    print(f'Generating visualizations for TSP-{args.n}...')
    print(f'Output directory: {args.output_dir}/')
    print()

    visualize_how_to_read(points, os.path.join(args.output_dir, '00_what_is_tsp.png'))
    visualize_nn(points, os.path.join(args.output_dir, '01_nearest_neighbor.png'))
    visualize_christofides(points, os.path.join(args.output_dir, '02_christofides.png'))
    visualize_2opt(points, os.path.join(args.output_dir, '03_two_opt.png'))
    visualize_comparison(points, os.path.join(args.output_dir, '04_comparison.png'))

    print(f'\nDone! {5} figures saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
