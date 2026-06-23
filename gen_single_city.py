"""Generate 6 individual city delivery figures, one per method."""

import sys, os, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_p = os.path.dirname(__file__)
_out = os.path.join(_p, 'outputs')
sys.path.append(os.path.join(_p, 'src'))
from src.utils import compute_distance_matrix, tour_cost
from city_delivery_scenario import generate_city_scenario

pts = generate_city_scenario(500, seed=42)
dist_mat = compute_distance_matrix(pts)

with open(os.path.join(_p, 'outputs', '_city_tours.pkl'), 'rb') as f:
    tours = pickle.load(f)

hoods = [
    ('North',(0.35,0.65,0.30,0.25),'lightcoral'),
    ('East',(0.70,0.45,0.15,0.20),'lightblue'),
    ('South',(0.30,0.05,0.25,0.20),'lightgreen'),
    ('West',(0.05,0.40,0.20,0.30),'lightyellow'),
    ('Biz',(0.40,0.45,0.10,0.15),'plum'),
]

methods = [
    ('NN', 'Nearest Neighbor'),
    ('Christofides', 'Christofides'),
    ('C+2opt', 'C+2opt'),
    ('DIFUSCO', 'DIFUSCO+2opt'),
    ('DualOpt', 'DualOpt'),
    ('LKH3', 'LKH3'),
]

for key, name in methods:
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Background zones
    for _, rc, cl in hoods:
        ax.add_patch(plt.Rectangle((rc[0],rc[1]), rc[2], rc[3],
                     fill=True, facecolor=cl, alpha=0.1,
                     edgecolor='gray', linewidth=0.3))

    # Points
    ax.scatter(pts[1:,0], pts[1:,1], c='steelblue', s=3, alpha=0.5, zorder=2)
    ax.scatter(pts[0,0], pts[0,1], c='red', s=150, marker='*',
               edgecolors='darkred', linewidth=2, zorder=5)

    # Tour edges
    tour = tours[key]
    for i in range(len(tour)-1):
        ax.plot([pts[tour[i],0], pts[tour[i+1],0]],
                [pts[tour[i],1], pts[tour[i+1],1]],
                '-', color='darkblue', linewidth=0.25, alpha=0.4)

    cost = tour_cost(dist_mat, tour)
    ax.set_title('%s\nCost: %.2f' % (name, cost), fontsize=18, fontweight='bold', pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')

    fname = 'city_%s.png' % key.lower().replace('+','')
    fig.savefig(os.path.join(_out, fname), dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(fname)

print('Done: 6 individual figures saved')
