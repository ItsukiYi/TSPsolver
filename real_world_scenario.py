r"""Real-World Scenario: Campus Food Delivery Route Optimization.

Scenario:
  A delivery rider at "Campus Eats" central kitchen must deliver 30 orders
  to dormitories and academic buildings across a university campus before
  the food gets cold. The campus has clustered dorm areas, a central academic
  quad, and scattered facilities (library, gym, admin).

  Goal: Find the shortest route that visits all 30 locations and returns to
  the kitchen. Compare all algorithms from our study.

The coordinates are designed to be realistic:
  - East Dorms (Cluster A): 10 buildings in a tight grid
  - West Dorms (Cluster B): 8 buildings along a curved road
  - Academic Quad (Cluster C): 7 buildings in a ring
  - Scattered facilities: 5 buildings spread across campus
"""

import sys, os, time, json, pickle, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import pdist, squareform

_project = os.path.dirname(__file__)
_venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')
sys.path.append(os.path.join(_project, 'src'))


def generate_campus_scenario(seed=42):
    """Generate realistic campus delivery coordinates."""
    np.random.seed(seed)
    points = []

    # Depot: Central Kitchen (center of campus)
    depot = np.array([0.50, 0.52])

    # Cluster A: East Dorms (10 buildings, tight grid)
    for i in range(4):
        for j in range(3):
            if len(points) < 10:
                x = 0.72 + i * 0.04 + np.random.randn() * 0.008
                y = 0.65 + j * 0.06 + np.random.randn() * 0.008
                points.append(np.clip([x, y], 0, 1))

    # Cluster B: West Dorms (8 buildings, along a curved road)
    for i in range(8):
        angle = np.pi * 0.6 + i * np.pi * 0.08
        r = 0.18 + np.random.randn() * 0.01
        x = 0.25 + r * np.cos(angle)
        y = 0.55 + r * np.sin(angle)
        points.append(np.clip([x, y], 0, 1))

    # Cluster C: Academic Quad (7 buildings, ring layout)
    for i in range(7):
        angle = 2 * np.pi * i / 7 + np.random.randn() * 0.05
        r = 0.10 + np.random.randn() * 0.01
        x = 0.50 + r * np.cos(angle)
        y = 0.28 + r * np.sin(angle)
        points.append(np.clip([x, y], 0, 1))

    # Scattered: Library, Gym, Admin, Sports Center, Health Center
    scattered = [
        [0.45, 0.45],   # Library (near center)
        [0.68, 0.18],   # Sports Center (south)
        [0.15, 0.72],   # Admin (northwest)
        [0.55, 0.80],   # Health Center (north)
        [0.38, 0.15],   # Parking (south)
    ]
    for pt in scattered:
        pt = np.array(pt) + np.random.randn(2) * 0.01
        points.append(np.clip(pt, 0, 1))

    # Combine: depot is point 0
    all_points = np.vstack([depot.reshape(1, 2), np.array(points)])
    return all_points


def solve_with_lkh(pts):
    """Solve TSP using LKH3."""
    import lkh, tsplib95
    os.environ['PATH'] = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']
    n = len(pts)
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'; problem.type = 'TSP'; problem.dimension = n
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {i+1: (float(pts[i][0]*1e6), float(pts[i][1]*1e6)) for i in range(n)}
    sol = lkh.solve('LKH.exe', problem=problem, max_trials=100, runs=10)
    tour = [r-1 for r in sol[0]]
    if tour[0] != tour[-1]: tour.append(tour[0])
    return tour


def solve_with_difusco(pts, ckpt):
    """DIFUSCO + 2-opt via subprocess."""
    script = '''
import sys, os, json, numpy as np, torch
_p = r"''' + _project + r'''"
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
model = TSPModel.load_from_checkpoint(r"''' + ckpt_path + r'''", param_args=args, strict=False)
model = model.to(device); model.eval()
pts = np.array(''' + pts_json + r''', dtype=np.float64); n = len(pts)
with torch.no_grad():
    pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
    ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
    for i in range(50):
        t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
        xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
    hm = xt.float().cpu().numpy().squeeze() + 1e-6
np_pts = pts.astype(np.float64)
tours, _ = merge_tours(hm[np.newaxis,:,:], np_pts, None, sparse_graph=False, parallel_sampling=1)
solved, _ = batched_two_opt_torch(np_pts, np.array(tours).astype('int64'), max_iterations=1000, device=device)
print(json.dumps(solved[0].tolist()))
'''
    r = subprocess.run([_venv_py, '-c', script], capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout.strip().split('\n')[-1])


def solve_with_dualopt(pts):
    """DualOpt (sliding window reviser only)."""
    sys.path.insert(0, os.path.join(_project, 'DualOpt-main'))
    from utils import load_model
    from utils.functions import LCP_TSP, load_problem
    import torch

    device = torch.device('cuda')
    gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
    revisers = []
    for sz in [50, 20, 10]:
        path = os.path.join(_project, 'DualOpt-main', 'pretrained', 'local_%d' % sz, 'epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    from src.algorithms import christofides_with_2opt
    from src.utils import compute_distance_matrix, tour_cost
    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=100)
    it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
    for rid in range(3):
        rlen = [50,20,10][rid]
        if rlen < seeds.shape[1]:  # skip if window > instance size
            seeds = LCP_TSP(seeds, gc, revisers[rid], rlen, [25,10,5][rid])
    # Map back to vertex indices
    from scipy.spatial import KDTree
    coords = seeds[0].cpu().numpy()
    tree = KDTree(pts)
    _, indices = tree.query(coords)
    tour = indices.tolist()
    if tour[0] != tour[-1]: tour.append(tour[0])
    return tour


# ---- Main ----
print('=' * 65)
print('REAL-WORLD SCENARIO: Campus Food Delivery Optimization')
print('=' * 65)

# Generate scenario
points = generate_campus_scenario(seed=42)
n = len(points)
print(f'\nCampus Map: {n} locations (1 kitchen + {n-1} delivery points)')
print('  Cluster A: East Dorms (10 buildings)')
print('  Cluster B: West Dorms (8 buildings)')
print('  Cluster C: Academic Quad (7 buildings)')
print('  Scattered: Library, Gym, Admin, Sports Center, Health Center')

# Get DIFUSCO ckpt
import glob
ckpt = sorted(glob.glob(os.path.join(_project, 'tsp50_categorical/checkpoints/epoch=6-step*.ckpt')))[0]

# Run all algorithms
results = {}
methods = {}

# 1. Nearest Neighbor
from src.algorithms import nearest_neighbor_tsp, christofides_tsp, christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost
t0 = time.time()
tour, _ = nearest_neighbor_tsp(points)
results['Nearest Neighbor'] = {'tour': tour, 'time': time.time()-t0}

# 2. Christofides
t0 = time.time()
tour, _ = christofides_tsp(points)
results['Christofides'] = {'tour': tour, 'time': time.time()-t0}

# 3. Christofides + 2-opt
t0 = time.time()
tour, _ = christofides_with_2opt(points, max_2opt_iterations=1000)
results['C+2opt'] = {'tour': tour, 'time': time.time()-t0}

# 4. LKH3
print('\n[LKH3] Solving...')
t0 = time.time()
tour = solve_with_lkh(points)
results['LKH3'] = {'tour': tour, 'time': time.time()-t0}

# 5. DIFUSCO + 2-opt
pts_json = json.dumps(points.tolist())
ckpt_path = ckpt.replace('\\', '/')
print('[DIFUSCO] Solving...')
t0 = time.time()
try:
    tour = solve_with_difusco(points, ckpt_path)
    results['DIFUSCO+2opt'] = {'tour': tour, 'time': time.time()-t0}
except Exception as e:
    print('  DIFUSCO failed:', e)
    results['DIFUSCO+2opt'] = {'tour': results['C+2opt']['tour'], 'time': 0, 'failed': True}

# 6. DualOpt
print('[DualOpt] Solving...')
t0 = time.time()
try:
    tour = solve_with_dualopt(points)
    results['DualOpt'] = {'tour': tour, 'time': time.time()-t0}
except Exception as e:
    print('  DualOpt failed:', e)
    results['DualOpt'] = {'tour': results['C+2opt']['tour'], 'time': 0, 'failed': True}

# Compute costs
dist_mat = compute_distance_matrix(points)
for name in results:
    results[name]['cost'] = tour_cost(dist_mat, results[name]['tour'])

# ---- Summary Table ----
print('\n' + '=' * 65)
print('RESULTS: Campus Food Delivery (30 locations)')
print('=' * 65)
print(f'{"Algorithm":<22s} {"Distance":>10s} {"Time":>8s} {"vs Best":>8s}')
print('-' * 50)
best_cost = min(r['cost'] for r in results.values())
for name in ['Nearest Neighbor', 'Christofides', 'C+2opt', 'DIFUSCO+2opt', 'DualOpt', 'LKH3']:
    if name in results:
        r = results[name]
        gap = (r['cost']/best_cost - 1)*100
        t_str = '%.1fs' % r['time'] if r['time'] < 60 else '%.1fm' % (r['time']/60)
        marker = ' < BEST' if r['cost'] <= best_cost + 1e-6 else ''
        fail = ' (FAILED)' if r.get('failed') else ''
        print(f'{name:<22s} {r["cost"]:10.2f}m {t_str:>8s} {gap:+7.1f}%{marker}{fail}')

# ---- Visualization ----
fig, axes = plt.subplots(2, 3, figsize=(20, 14))
plot_order = ['Nearest Neighbor', 'Christofides', 'C+2opt', 'DIFUSCO+2opt', 'DualOpt', 'LKH3']

# Campus background features
campus_zones = {
    'East Dorms': ([0.68, 0.58, 0.24, 0.16], 'lightcoral'),
    'West Dorms': ([0.10, 0.45, 0.30, 0.22], 'lightblue'),
    'Academic Quad': ([0.38, 0.15, 0.24, 0.24], 'lightgreen'),
    'Central Kitchen': ([0.47, 0.49, 0.06, 0.06], 'yellow'),
}

for idx, name in enumerate(plot_order):
    ax = axes[idx // 3, idx % 3]
    if name not in results: continue
    r = results[name]
    tour = r['tour']

    # Draw campus zones
    for zone_name, (rect, color) in campus_zones.items():
        rect_patch = plt.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                    fill=True, facecolor=color, alpha=0.2, edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect_patch)
        if rect[2] > 0.1:
            ax.text(rect[0]+rect[2]/2, rect[1]+rect[3]/2, zone_name, fontsize=6,
                   ha='center', va='center', alpha=0.5)

    # Draw tour
    for i in range(len(tour)-1):
        ax.plot([points[tour[i], 0], points[tour[i+1], 0]],
                [points[tour[i], 1], points[tour[i+1], 1]],
                '-', color='darkblue', linewidth=1.2, alpha=0.7)

    # Points
    ax.scatter(points[1:, 0], points[1:, 1], c='steelblue', s=40, edgecolors='black', linewidth=0.5, zorder=5)
    # Depot
    ax.scatter(points[0, 0], points[0, 1], c='red', s=150, marker='*', edgecolors='darkred', linewidth=1.5, zorder=10, label='Kitchen')
    # Annotate depot
    ax.annotate('Kitchen', (points[0, 0], points[0, 1]), fontsize=8, fontweight='bold',
               xytext=(5, 5), textcoords='offset points')

    ax.set_title('%s\n%.2fm | %.1fs' % (name, r['cost'], r['time']), fontsize=11, fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal'); ax.axis('off')

plt.suptitle('Campus Food Delivery Route Optimization\n30 Locations: Dormitories + Academic Buildings + Facilities',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
outpath = os.path.join(_project, 'outputs', 'campus_delivery_scenario.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print('\nSaved:', outpath)

# Save results
with open(os.path.join(_project, 'outputs', 'campus_scenario_results.json'), 'w') as f:
    json.dump({name: {'cost': float(r['cost']), 'time': float(r['time']),
                      'tour': [int(v) for v in r['tour']]}
               for name, r in results.items()}, f, indent=2)
