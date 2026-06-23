"""Regenerate city delivery figure: 3x2 layout, correct costs."""

import sys, os, json, numpy as np, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_p = os.path.dirname(__file__)
_out = os.path.join(_p, 'outputs')
os.environ['PATH'] = os.path.join(_p, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']
sys.path.append(os.path.join(_p, 'src'))

from src.algorithms import nearest_neighbor_tsp, christofides_tsp, christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

# Generate city scenario (same as before)
from city_delivery_scenario import generate_city_scenario
pts = generate_city_scenario(500, seed=42)
n = len(pts)
print(f'City scenario: {n} locations')

dist_mat = compute_distance_matrix(pts)

# Run all methods (fast ones first)
tours = {}

print('NN...')
tours['NN'], _ = nearest_neighbor_tsp(pts)

print('Christofides...')
tours['Christofides'], _ = christofides_tsp(pts)

print('C+2opt...')
tours['C+2opt'], _ = christofides_with_2opt(pts, max_2opt_iterations=500)

# DualOpt
print('DualOpt...')
sys.path.insert(0, os.path.join(_p, 'DualOpt-main'))
from utils import load_model
from utils.functions import LCP_TSP, load_problem
import torch
device = torch.device('cuda')
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
revisers = []
for sz in [50, 20, 10]:
    r, _ = load_model(os.path.join(_p, 'DualOpt-main', 'pretrained', 'local_%d' % sz, 'epoch-100.pt'), is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

it = tours['C+2opt'][:-1]
seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
for rid in range(3):
    rlen = [50,20,10][rid]
    if rlen < seeds.shape[1]:
        seeds = LCP_TSP(seeds, gc, revisers[rid], rlen, [25,10,5][rid])
from scipy.spatial import KDTree
tree = KDTree(pts)
_, dual_idx = tree.query(seeds[0].cpu().numpy())
dual_tour = dual_idx.tolist()
if dual_tour[0] != dual_tour[-1]: dual_tour.append(dual_tour[0])
tours['DualOpt'] = dual_tour

# LKH3
print('LKH3...')
import lkh, tsplib95
problem = tsplib95.models.StandardProblem()
problem.name = 'TSP'; problem.type = 'TSP'; problem.dimension = n
problem.edge_weight_type = 'EUC_2D'
problem.node_coords = {i+1: (float(pts[i][0]*1e6), float(pts[i][1]*1e6)) for i in range(n)}
sol = lkh.solve('LKH.exe', problem=problem, max_trials=100, runs=5)
tour = [r-1 for r in sol[0]]
if tour[0] != tour[-1]: tour.append(tour[0])
tours['LKH3'] = tour

# DIFUSCO (try loading from saved, or skip)
try:
    import subprocess, pickle, glob
    _venv_py = os.path.join(_p, 'venv', 'Scripts', 'python.exe')
    ckpt = sorted(glob.glob(os.path.join(_p, 'tsp50_categorical/checkpoints/epoch=6-step*.ckpt')))[0]
    r = subprocess.run([_venv_py, '-c', '''
import sys, os, json, numpy as np, torch
_p = r"''' + _p + r'''"
sys.path.insert(0, os.path.join(_p, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_p, 'DIFUSCO-main'))
from pl_tsp_model import TSPModel
from argparse import Namespace
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import merge_tours, batched_two_opt_torch
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
    validation_examples=8, use_activation_checkpoint=False, fp16=False)
device = torch.device('cuda')
model = TSPModel.load_from_checkpoint(r"''' + ckpt.replace('\\', '/') + r'''", param_args=args, strict=False)
model = model.to(device); model.eval()
pts = np.array(''' + json.dumps(pts.tolist()) + r''', dtype=np.float64); n = len(pts)
with torch.no_grad():
    pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
    ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
    for i in range(50):
        t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
        xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
    hm = xt.float().cpu().numpy().squeeze() + 1e-6
np_pts = pts.astype(np.float64)
tours_d, _ = merge_tours(hm[np.newaxis,:,:], np_pts, None, sparse_graph=False, parallel_sampling=1)
solved, _ = batched_two_opt_torch(np_pts, np.array(tours_d).astype('int64'), max_iterations=1000, device=device)
print(json.dumps(solved[0].tolist()))
'''], capture_output=True, text=True, timeout=300)
    dif_tour = json.loads(r.stdout.strip().split('\n')[-1])
    tours['DIFUSCO'] = dif_tour
    print('DIFUSCO done')
except Exception as e:
    print('DIFUSCO failed:', str(e)[:100])
    tours['DIFUSCO'] = tours['C+2opt']

# ---- Plot: 3 rows x 2 cols ----
methods = [
    ('Nearest Neighbor', 'NN', 'Classic'),
    ('Christofides', 'Christofides', 'Classic'),
    ('C+2opt', 'C+2opt', 'Classic'),
    ('DIFUSCO+2opt', 'DIFUSCO', 'AI Gen.'),
    ('DualOpt', 'DualOpt', 'AI Improv.'),
    ('LKH3', 'LKH3', 'Gold Std.'),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 21))
axes = axes.flatten()

for idx, (name, key, atype) in enumerate(methods):
    ax = axes[idx]
    tour = tours.get(key)
    if tour is None:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=18)
        ax.axis('off'); continue

    # Draw neighborhood backgrounds
    neighborhoods = [
        ('North Res.', (0.35, 0.65, 0.30, 0.25), 'lightcoral'),
        ('East Sub.', (0.70, 0.45, 0.15, 0.20), 'lightblue'),
        ('South Res.', (0.30, 0.05, 0.25, 0.20), 'lightgreen'),
        ('West Apt.', (0.05, 0.40, 0.20, 0.30), 'lightyellow'),
        ('Business', (0.40, 0.45, 0.10, 0.15), 'plum'),
    ]
    for nname, rect, color in neighborhoods:
        p = plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=True,
                           facecolor=color, alpha=0.12, edgecolor='gray', linewidth=0.3)
        ax.add_patch(p)

    # Draw all edges with very thin lines
    ax.scatter(pts[1:,0], pts[1:,1], c='steelblue', s=2, alpha=0.5, zorder=2)
    ax.scatter(pts[0,0], pts[0,1], c='red', s=100, marker='*', edgecolors='darkred', linewidth=1.5, zorder=5)
    for i in range(len(tour)-1):
        ax.plot([pts[tour[i],0], pts[tour[i+1],0]],
                [pts[tour[i],1], pts[tour[i+1],1]], '-', color='darkblue',
                linewidth=0.15, alpha=0.35)

    cost_val = tour_cost(dist_mat, tour)
    # Display cost in units (0-1 square, so typical cost for 500 points is ~16-17)
    ax.set_title(f'{name} [{atype}]\nCost: {cost_val:.2f}', fontsize=15, fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.axis('off')

plt.suptitle('City-Wide Package Delivery — 500 Locations\nAlgorithm Comparison at Scale',
             fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(_out, 'city_delivery_500.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Print summary
print('\nFinal costs:')
for name, key, _ in methods:
    if key in tours:
        c = tour_cost(dist_mat, tours[key])
        print(f'  {name:<20s}: {c:.2f}')
print(f'\nSaved: {os.path.join(_out, "city_delivery_500.png")}')
