r"""Architecture diagrams for DIFUSCO, DualOpt, and the pipeline."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

_out = os.path.join(os.path.dirname(__file__), 'outputs')

def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=16, bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=weight)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.5):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ============================================================
# Figure 1: DIFUSCO Architecture
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
ax.set_title('DIFUSCO: Graph-based Diffusion Solver for TSP', fontsize=16, fontweight='bold', pad=20)

# Input
draw_box(ax, 1, 3.5, 2, 1, 'Node Coordinates\n$(x_i, y_i)_{i=1}^n$', 'lightyellow', 10, True)
# Noise
draw_box(ax, 1, 1.5, 2, 1, 'Random Noise\n$\\mathbf{x}_T \\sim \\text{Bernoulli}(0.5)$', 'lightcoral', 9)

# GNN Encoder
draw_box(ax, 4.5, 2.5, 3, 2.2, 'Anisotropic Gated GNN\n(12 layers, d=256)\n\n$\\hat{x}_0 = f_\\theta(x_t, \\text{coord}, t)$\nDenoise for 50 steps', 'lightblue', 9)
draw_arrow(ax, 2, 3.2, 3, 3.2)
draw_arrow(ax, 2, 1.8, 3, 2.2)

# Heatmap
draw_box(ax, 8, 2.5, 2.2, 1.2, 'Edge Heatmap\n$H_{ij} \\in [0,1]$', 'lightgreen', 10, True)
draw_arrow(ax, 6, 2.5, 6.9, 2.5)

# Greedy Merge
draw_box(ax, 10.5, 2.5, 2.2, 1.2, 'Greedy Merge\nrank by $H_{ij}/d(i,j)$', 'plum', 9)
draw_arrow(ax, 9.1, 2.5, 9.4, 2.5)

# 2-opt
draw_box(ax, 13, 3.5, 1.8, 0.8, '2-opt\nPolish', 'orange', 10, True)
draw_arrow(ax, 11.6, 2.8, 12.1, 3.3)

# Output
draw_box(ax, 13, 1.5, 2, 1, 'Valid TSP Tour\n$\\pi = (v_1,\\dots,v_n)$', 'lightyellow', 10, True)
draw_arrow(ax, 11.6, 2.2, 12.1, 1.7)

# Labels
ax.text(2, 4.5, 'Input', ha='center', fontsize=8, color='gray')
ax.text(4.5, 4.7, 'Denoising Process', ha='center', fontsize=8, color='gray')
ax.text(8, 4, 'Decoding', ha='center', fontsize=8, color='gray')

plt.tight_layout()
fig.savefig(os.path.join(_out, 'arch_difusco.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: arch_difusco.png')


# ============================================================
# Figure 2: DualOpt Architecture
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.set_xlim(0, 15); ax.set_ylim(0, 5); ax.axis('off')
ax.set_title('DualOpt: Dual Divide-and-Optimize Algorithm', fontsize=16, fontweight='bold', pad=20)

# Phase 1: Grid D&C
ax.text(3.75, 4.7, 'Phase 1: Grid Divide-and-Conquer', ha='center', fontsize=11, fontweight='bold', color='darkblue')

draw_box(ax, 0.8, 3.5, 1.6, 1, 'Input\nCoordinates', 'lightyellow', 9)
draw_arrow(ax, 1.6, 3.5, 2.3, 3.5)
draw_box(ax, 3.5, 3.5, 2, 1, 'Grid Partition\n$M \\times M$ cells', 'lightblue', 9)
draw_arrow(ax, 4.5, 3.5, 5.5, 3.5)
draw_box(ax, 7, 3.5, 2.4, 1, 'LKH3 per Cell\n(parallel)', 'lightgreen', 10, True)
draw_arrow(ax, 8.2, 3.5, 9, 3.5)
draw_box(ax, 10.2, 3.5, 1.8, 1, 'Merge Cells\nIteratively', 'plum', 9)
draw_arrow(ax, 11.1, 3.5, 12, 3.5)
draw_box(ax, 13.5, 3.5, 1.5, 1, 'Initial\nTour', 'orange', 10, True)

# Phase 2: Path-based Reviser
ax.text(7.5, 2.2, 'Phase 2: Path-based Neural Refinement', ha='center', fontsize=11, fontweight='bold', color='darkred')

draw_box(ax, 1, 0.8, 2.2, 1.2, 'Sliding Windows\n$k=50 \\to 20 \\to 10$', 'lightcoral', 9)
draw_arrow(ax, 2.1, 0.8, 3.5, 0.8)
draw_box(ax, 5.2, 0.8, 2.6, 1.2, 'Neural Reviser\n(Attention Model)\nTrained via REINFORCE', 'lightblue', 9)
draw_arrow(ax, 6.5, 0.8, 8, 0.8)
draw_box(ax, 9.5, 0.8, 2.2, 1.2, 'Improved\nSub-tours', 'lightgreen', 10, True)
draw_arrow(ax, 10.6, 0.8, 12, 0.8)
draw_box(ax, 13.5, 0.8, 1.5, 1, 'Refined\nTour', 'orange', 10, True)

# Iteration arrow
draw_arrow(ax, 14.8, 1.8, 14.8, 2.8)
ax.text(14.3, 2.3, 'repeat', fontsize=7, color='gray', rotation=90)

plt.tight_layout()
fig.savefig(os.path.join(_out, 'arch_dualopt.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: arch_dualopt.png')


# ============================================================
# Figure 3: DIFUSCO -> DualOpt Pipeline
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis('off')
ax.set_title('Improvement #2: DIFUSCO to DualOpt Hybrid Pipeline', fontsize=16, fontweight='bold', pad=20)

# Phase 1
ax.text(2, 4.2, 'Phase 1: Diffusion Construction (DIFUSCO)', ha='center', fontsize=11, fontweight='bold', color='darkblue')

draw_box(ax, 0.5, 2.8, 1.5, 1, 'Coordinates\n$\\mathbb{R}^{n\\times 2}$', 'lightyellow', 9)
draw_arrow(ax, 1.25, 2.8, 1.25, 2.1)
draw_box(ax, 2.8, 1.5, 2.5, 1.5, 'GNN Denoising\n50 steps, Categorical\n$\\mathbf{x}_T \\to \\hat{H}$', 'lightblue', 9)
draw_arrow(ax, 4.05, 2, 5.2, 2)
draw_box(ax, 6.3, 1.5, 2, 1.5, 'Greedy Merge\nrank $H_{ij}/d(i,j)$\n$\\to$ initial tour $\\pi_0$', 'lightgreen', 9)
draw_arrow(ax, 7.3, 1.5, 8.5, 1.5)
draw_box(ax, 9.6, 1.5, 1.8, 1.5, 'Optional\n2-opt Polish\n(1000 iters)', 'plum', 8)

# Phase 2
ax.text(10.5, 4.2, 'Phase 2: Neural Refinement (DualOpt)', ha='center', fontsize=11, fontweight='bold', color='darkred')

draw_arrow(ax, 10.5, 2.25, 10.5, 1.8)
draw_box(ax, 10.5, 1.1, 1.8, 0.8, 'Reviser k=50\n(25 iters)', 'lightcoral', 8)
draw_arrow(ax, 11.4, 1.1, 12.2, 1.1)
draw_box(ax, 13, 1.1, 1.8, 0.8, 'Reviser k=20\n(10 iters)', 'lightcoral', 8)
draw_arrow(ax, 13, 0.3, 13, 1.2)  # wrap

# Output
draw_arrow(ax, 13.9, 0.7, 13.9, 0.4)
draw_box(ax, 11.5, 0.1, 2, 0.7, 'Final Tour $\\pi^*$', 'orange', 10, True)

# Comparison
ax.text(7, 3.8, 'vs C+2opt baseline: $-1.67$% (TSP-50) | $+1.56$% (TSP-100, per-scale training)', ha='center', fontsize=10, color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
fig.savefig(os.path.join(_out, 'arch_pipeline.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: arch_pipeline.png')


# ============================================================
# Figure 4: Algorithm Process Overview (all methods side by side)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

methods = [
    ('Nearest Neighbor', [
        'Start at depot',
        'Find nearest\nunvisited node',
        'Move there,\nmark visited',
        'Repeat $n-1$ times',
        'Return to depot',
    ], 'Greedy construction\n$O(n^2)$'),
    ('Christofides', [
        'Build MST\n(Prim, $O(n^2)$)',
        'Find odd-degree\nvertices',
        'Min-weight perfect\nmatching (Blossom)',
        'Combine MST+matching\n$\\to$ Eulerian graph',
        'Eulerian circuit\n$\\to$ shortcut $\\to$ tour',
    ], '1.5-approximation\n$O(n^3)$'),
    ('2-opt Local Search', [
        'Start with any\ntour (e.g., NN)',
        'Try all pairs\n$(i,j)$ of edges',
        'If $d(i,j)+d(i+1,j+1)$\n$< d(i,i+1)+d(j,j+1)$',
        'Swap: reverse\nsegment $[i+1, j]$',
        'Repeat until\nno improvement',
    ], 'Local improvement\n$O(n^2)$ per iter'),
    ('DIFUSCO', [
        'Random noise\n$\\mathbf{x}_T$ (Bernoulli)',
        'GNN: 50 denoising\nsteps $\\to$ heatmap $H$',
        'Greedy merge:\nrank $H_{ij}/d(i,j)$',
        'Extract valid\ntour from heatmap',
        '2-opt polish\n(1000 iterations)',
    ], 'Generative (AI)\n$O(n^2\\cdot d^2)$'),
    ('DualOpt (Reviser)', [
        'Start with C+2opt\ninitial tour',
        'Sliding window\n$k=50$: coarse revise',
        'Sliding window\n$k=20$: medium revise',
        'Sliding window\n$k=10$: fine polish',
        'Neural policy trained\nvia REINFORCE',
    ], 'Improvement (AI)\n$O(n\\cdot k^2)$'),
    ('LKH3', [
        'Start with any\ntour',
        '$k$-opt search\nguided by $\\alpha$-nearness',
        '$\\alpha$ from minimum\n1-tree (MST lower bound)',
        'Try $k=2,3,4,5$ opt\nmoves simultaneously',
        'Pure C implementation\nhighly optimized',
    ], 'Industry standard\n$\\approx O(n^{2.2})$'),
]

for idx, (name, steps, note) in enumerate(methods):
    ax = axes[idx // 3, idx % 3]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title(name, fontsize=12, fontweight='bold')

    colors = ['lightyellow', 'lightblue', 'lightgreen', 'plum', 'orange']
    for i, step in enumerate(steps):
        y = 8.5 - i * 1.6
        box = FancyBboxPatch((0.5, y-0.5), 9, 1.3, boxstyle="round,pad=0.1",
                              facecolor=colors[i % len(colors)], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, y+0.15, step, ha='center', va='center', fontsize=8)
        if i < len(steps) - 1:
            ax.annotate('', xy=(5, y-0.6), xytext=(5, y-1.0),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Note at bottom
    ax.text(5, 0.3, note, ha='center', fontsize=9, fontstyle='italic', color='darkblue',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.suptitle('TSP Algorithm Processes: Classic vs Modern Methods', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(_out, 'algorithm_processes.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: algorithm_processes.png')


# ============================================================
# Figure 5: Improvement Strategies Overview
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 9))

improvements = [
    ('#1: Heatmap-Guided Reviser', [
        'DIFUSCO $\\to$ heatmap $H$',
        'Compute edge confidence\n$c_i = H_{\\pi_i,\\pi_{i+1}}$',
        'For each reviser window:\nskip if avg($c_w$) > $\\tau$',
        'Only revise\nuncertain windows',
    ], 'Result: 0.00% (TSP-50)\nHeatmap signal too noisy'),
    ('#2: DIFUSCO $\\to$ DualOpt Pipeline', [
        'DIFUSCO: noise $\\to$ heatmap',
        'Greedy merge $\\to$ initial tour',
        'DualOpt revisers\n$k=50\\to20\\to10$ refine',
        'Two-stage:\ngeneration + improvement',
    ], 'Result: $-$1.67% (TSP-50)\nBest improvement!'),
    ('#3: Adaptive Window Sizing', [
        'Run 2-opt diagnostic\n(10 iters, GPU)',
        'Track changed edges\n$\\to$ instability mask',
        'Allocate more reviser\niters to unstable windows',
        'Skip stable windows',
    ], 'Result: +0.74% (TSP-50)\n2-opt too conservative'),
    ('#4: Fragment Freezing', [
        'Run DIFUSCO + C+2opt\nindependently',
        'Find edge intersection\n$E_{\\text{dif}} \\cap E_{\\text{c2}}$',
        'Freeze agreed edges\nlet reviser handle rest',
        'Consensus = confidence',
    ], 'Result: +1.51% mean\n4/10 instances improved'),
    ('#5: Destroy-and-Repair', [
        'DIFUSCO heatmap $\\to$\nedge confidence scores',
        'Destroy $K$ lowest-\nconfidence edges',
        'Break tour into $K$\nconnected segments',
        'Greedy reconnect\n+ 2-opt polish',
    ], 'Result: 0.00% (fixed bug)\nDualOpt already optimal'),
    ('#6: LKH3 Polish', [
        'DualOpt output tour',
        'Feed to LKH3 as\ninitial solution',
        'LKH3 runs with\nthis warm start',
        'Check if LKH3 finds\nfurther improvements',
    ], 'Result: 0.00%\nDualOpt = LKH3-local-opt'),
]

for idx, (name, steps, note) in enumerate(improvements):
    ax = axes[idx // 3, idx % 3]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title(name, fontsize=11, fontweight='bold')

    color = 'lightgreen' if 'Best' in note or 'improved' in note.lower() else \
            'lightyellow' if '0.00%' in note else 'lightcoral'
    colors_list = ['lightyellow', 'lightblue', 'plum', 'lightsalmon']
    for i, step in enumerate(steps):
        y = 8.5 - i * 1.6
        box = FancyBboxPatch((0.5, y-0.5), 9, 1.3, boxstyle="round,pad=0.1",
                              facecolor=colors_list[i], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(5, y+0.15, step, ha='center', va='center', fontsize=8)
        if i < len(steps) - 1:
            ax.annotate('', xy=(5, y-0.6), xytext=(5, y-1.0),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(5, 0.3, note, ha='center', fontsize=9, fontstyle='italic', color='darkblue',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

plt.suptitle('Six Improvement Strategies: Design and Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(_out, 'improvement_strategies.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved: improvement_strategies.png')

print('\nAll architecture diagrams saved to', _out)
