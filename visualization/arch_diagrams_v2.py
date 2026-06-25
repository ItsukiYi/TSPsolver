r"""Clean, readable architecture diagrams with large fonts."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

_out = os.path.join(os.path.dirname(__file__), 'outputs')
FS = 18  # base font size for box text

def add_box(ax, x, y, w, h, text, color='lightblue', fs=FS):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold')
    return box

def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))


# ============================================================
# Figure 1: DIFUSCO (simplified, 4 boxes, very large text)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis('off')

add_box(ax, 2, 3, 4, 2, 'Input\nNode Coordinates\n$(x_i, y_i)_{i=1}^n$', 'lightyellow', 20)
add_box(ax, 8, 3, 5, 2.5, 'GNN Denoising\n(50 steps, 12 layers)\nNoise $→$ Edge Heatmap $H_{ij}$', 'lightblue', 20)
add_box(ax, 15, 3, 4, 2, 'Decode\nGreedy Merge +\n2-opt Polish', 'lightgreen', 20)

add_arrow(ax, 4, 3, 5.5, 3)

# Label at bottom
add_box(ax, 17.5, 0.8, 3, 1, 'Valid TSP Tour', 'orange', 22)

add_arrow(ax, 15, 2, 16.5, 1.3)
add_arrow(ax, 8, 1.7, 13, 1.7)
# arrow from decode to output

ax.text(10, 5.5, 'DIFUSCO: Graph-based Diffusion Solver for TSP', ha='center', fontsize=28, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(_out, 'arch_difusco.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('arch_difusco.png')


# ============================================================
# Figure 2: DualOpt (2-phase, 5 boxes)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
ax.set_xlim(0, 20); ax.set_ylim(0, 7); ax.axis('off')

ax.text(5, 6.5, 'Phase 1: Grid Divide-and-Conquer', ha='center', fontsize=22, fontweight='bold', color='darkblue')
add_box(ax, 2, 5, 3, 1.5, 'Partition Plane\ninto $M \\times M$ Grid', 'lightyellow', 18)
add_box(ax, 7, 5, 3, 1.5, 'Solve Each Cell\nwith Parallel LKH3', 'lightblue', 18)
add_box(ax, 12, 5, 3, 1.5, 'Merge Grids\nHierarchically', 'lightgreen', 18)
add_box(ax, 17, 5, 3, 1.5, 'Initial\nTour', 'orange', 20)
add_arrow(ax, 3.5, 5, 5.5, 5)
add_arrow(ax, 8.5, 5, 10.5, 5)
add_arrow(ax, 13.5, 5, 15.5, 5)

ax.text(5, 3.2, 'Phase 2: Neural Reviser Refinement', ha='center', fontsize=22, fontweight='bold', color='darkred')
add_box(ax, 5, 1.5, 5, 2, 'Sliding Window Revisers\n$k=50$ (coarse) $→$ $k=20$ (medium) $→$ $k=10$ (fine)\nTrained via REINFORCE', 'lightcoral', 18)
add_box(ax, 14, 1.5, 3, 2, 'Refined\nTour', 'orange', 22)
add_arrow(ax, 7.5, 1.5, 12.5, 1.5)

# Down arrow from Phase 1 to Phase 2
add_arrow(ax, 17, 3.8, 17, 3.3)

ax.text(10, 6.8, 'DualOpt: Dual Divide-and-Optimize Algorithm', ha='center', fontsize=28, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(_out, 'arch_dualopt.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('arch_dualopt.png')


# ============================================================
# Figure 3: Pipeline (2-phase, 6 boxes)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(20, 7))
ax.set_xlim(0, 20); ax.set_ylim(0, 7); ax.axis('off')

ax.text(5, 6.5, 'Phase 1: Diffusion Construction (DIFUSCO)', ha='center', fontsize=22, fontweight='bold', color='darkblue')
add_box(ax, 2, 4.5, 3, 2, 'GNN Denoising\n50 steps\nNoise $→$ Heatmap', 'lightblue', 18)
add_box(ax, 8, 4.5, 3, 2, 'Greedy Merge\nHeatmap $→$\nInitial Tour $π_0$', 'lightgreen', 18)
add_arrow(ax, 3.5, 4.5, 6.5, 4.5)

ax.text(14, 6.5, 'Phase 2: Refinement (DualOpt)', ha='center', fontsize=22, fontweight='bold', color='darkred')
add_box(ax, 14, 4.5, 4, 2, '3 Revisers\n$k=50 → 20 → 10$\nSliding Windows', 'lightcoral', 18)
add_arrow(ax, 9.5, 4.5, 12, 4.5)

add_box(ax, 8, 1.5, 6, 1.5, 'Final Optimized Tour $π^*$\n$-1.67$% vs GT (TSP-50) | Best Improvement!', 'orange', 22)
add_arrow(ax, 14, 3.5, 10, 2.2)

# Comparison label
ax.text(2, 1.5, 'vs C+2opt baseline:\nDualOpt reviser extracts 24%\nmore improvement than 2-opt', ha='center',
        fontsize=14, color='darkgreen', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax.text(10, 6.8, 'Improvement #2: DIFUSCO → DualOpt Hybrid Pipeline', ha='center', fontsize=28, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(_out, 'arch_pipeline.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('arch_pipeline.png')


# ============================================================
# Figure 4: Algorithm comparison (6 methods, 1 row per method)
# ============================================================
methods = [
    ('Nearest Neighbor', ['1. Start at depot', '2. Visit nearest unvisited node', '3. Repeat until all visited', '4. Return to depot'], '$O(n^2)$ | Gap 20-30%'),
    ('Christofides', ['1. Build MST', '2. Min-weight perfect matching on odd-degree vertices', '3. Combine MST + matching → Eulerian graph', '4. Eulerian circuit → shortcut → tour'], '$O(n^3)$ | 1.5x guarantee | Gap 3-7%'),
    ('DIFUSCO (AI Gen.)', ['1. Random noise → 50-step GNN denoising → heatmap', '2. Greedy merge: rank edges by heatmap/distance', '3. Extract valid tour', '4. 2-opt polish'], '$O(n^2 d^2)$ | Trained on TSP-50'),
    ('DualOpt (AI Improv.)', ['1. Start with C+2opt initial tour', '2. Sliding window reviser k=50 → coarse', '3. Sliding window reviser k=20 → medium', '4. Sliding window reviser k=10 → fine polish'], '$O(n)$ | Constant time ~6s'),
    ('LKH3 (Gold Std.)', ['1. Start with any tour', '2. k-opt search guided by α-nearness (from min 1-tree)', '3. Try k=2,3,4,5 opt simultaneously', '4. Highly optimized C implementation'], '$O(n^{2.2})$ | Gap <0.5%'),
]

fig, axes = plt.subplots(len(methods), 1, figsize=(20, 14))
fig.subplots_adjust(hspace=0.5)

for idx, (name, steps, note) in enumerate(methods):
    ax = axes[idx]
    ax.set_xlim(0, 20); ax.set_ylim(0, 2); ax.axis('off')

    # Method name on left
    ax.text(0.3, 1, name, ha='left', va='center', fontsize=20, fontweight='bold')

    # Steps as connected boxes
    colors = ['lightyellow', 'lightblue', 'lightgreen', 'plum']
    for i, step in enumerate(steps):
        x = 4.5 + i * 4
        box = FancyBboxPatch((x-1.5, 0.4), 3, 1.2, boxstyle="round,pad=0.1",
                              facecolor=colors[i], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, 1, step, ha='center', va='center', fontsize=14, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x+1.6, 1), xytext=(x+2.3, 1),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Note on right
    ax.text(19.5, 1, note, ha='right', va='center', fontsize=14, fontstyle='italic', color='darkblue')

plt.suptitle('TSP Algorithm Processes: Classic vs AI Methods', fontsize=26, fontweight='bold', y=1.02)
fig.savefig(os.path.join(_out, 'algorithm_processes.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('algorithm_processes.png')


# ============================================================
# Figure 5: Improvement strategies (6 boxes, compact)
# ============================================================
improvements = [
    ('#1: Heatmap-Guided', 'Skip high-confidence\nreviser windows', '0.00%\n(heatmap too noisy)', 'lightcoral'),
    ('#2: DIFUSCO→DualOpt', 'Diffusion tour +\nneural refinement', '-1.67%\n(7/10 improved)', 'lightgreen'),
    ('#3: Adaptive Window', '2-opt diagnostic\nguides reviser', '+0.74%\n(diagnostic too weak)', 'lightcoral'),
    ('#4: Fragment Freezing', 'Solver consensus\nlocks edges', '+1.51% mean\n(4/10 improved!)', 'lightyellow'),
    ('#5: Destroy-Repair', 'Destroy uncertain edges\n+ greedy repair', '0.00% (fixed bug)\n(DualOpt optimal)', 'lightcoral'),
    ('#6: LKH3 Polish', 'LKH3 post-process\non DualOpt output', '0.00%\n(already optimal)', 'lightcoral'),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for idx, (name, idea, result, color) in enumerate(improvements):
    ax = axes[idx]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    add_box(ax, 5, 7.5, 9, 2, name, 'lightgray', 24)
    add_box(ax, 5, 4.5, 9, 3, f'Idea:\n{idea}', 'lightblue', 18)
    add_box(ax, 5, 1.5, 9, 2, f'Result:\n{result}', color, 18)

plt.suptitle('Six Improvement Strategies: Design \& Results', fontsize=28, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(_out, 'improvement_strategies.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('improvement_strategies.png')

print('\nAll done!')
