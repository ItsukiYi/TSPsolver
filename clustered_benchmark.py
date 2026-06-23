r"""Clustered TSP Benchmark — a dataset neither DIFUSCO nor DualOpt tested.

Generates realistic delivery scenarios: 3-5 neighborhood clusters + scattered
houses + central depot. This represents real last-mile delivery topology.

Compares all methods: NN, Christofides, C+2opt, DIFUSCO+2opt, DualOpt, LKH3.
"""

import sys, os, time, json, pickle, glob, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_project = os.path.dirname(__file__)
_venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')
os.environ['PATH'] = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']
sys.path.append(os.path.join(_project, 'src'))

from src.algorithms import nearest_neighbor_tsp, christofides_tsp, christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost


def generate_clustered_instance(n, n_clusters=4, seed=42):
    """Generate realistic clustered delivery instance.

    Args:
        n: total nodes (including depot)
        n_clusters: number of neighborhood clusters
        seed: random seed
    Returns:
        points: (n, 2) array, depot is index 0
    """
    np.random.seed(seed)
    # Depot at city center
    depot = np.array([0.50, 0.50])

    # Distribute remaining nodes across clusters + scattered
    nodes_per_cluster = (n - 1) * 3 // 4 // n_clusters  # 75% in clusters
    n_clustered = nodes_per_cluster * n_clusters
    n_scattered = n - 1 - n_clustered

    points = []
    cluster_centers = []
    for c in range(n_clusters):
        angle = 2 * np.pi * c / n_clusters + np.random.rand() * 0.3
        r = 0.20 + np.random.rand() * 0.15
        cx = 0.50 + r * np.cos(angle)
        cy = 0.50 + r * np.sin(angle)
        cluster_centers.append((cx, cy))

        # Cluster: gaussian around center with varying density
        cluster_radius = 0.04 + np.random.rand() * 0.06
        for _ in range(nodes_per_cluster):
            x = cx + np.random.randn() * cluster_radius
            y = cy + np.random.randn() * cluster_radius
            points.append(np.clip([x, y], 0, 1))

    # Scattered points across the map
    for _ in range(n_scattered):
        x = np.random.rand()
        y = np.random.rand()
        points.append(np.array([x, y]))

    return np.vstack([depot.reshape(1, 2), np.array(points)])


# ---- Heatmap generation via subprocess ----
print('Step 1: Generating DIFUSCO heatmaps for clustered instances...')
gen_code = r'''
import sys, os, pickle, numpy as np, torch
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
ckpt = os.path.join(_p, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

# Inline the generator function
def _gen(n, seed):
    np.random.seed(seed)
    depot = np.array([0.50, 0.50])
    n_clusters = 4
    nodes_per_cluster = (n - 1) * 3 // 4 // n_clusters
    n_scattered = n - 1 - nodes_per_cluster * n_clusters
    pts = []
    for c in range(n_clusters):
        angle = 2 * np.pi * c / n_clusters + np.random.rand() * 0.3
        r = 0.20 + np.random.rand() * 0.15
        cx = 0.50 + r * np.cos(angle); cy = 0.50 + r * np.sin(angle)
        cr = 0.04 + np.random.rand() * 0.06
        for _ in range(nodes_per_cluster):
            x = cx + np.random.randn() * cr; y = cy + np.random.randn() * cr
            pts.append(np.clip([x, y], 0, 1))
    for _ in range(n_scattered):
        pts.append(np.array([np.random.rand(), np.random.rand()]))
    return np.vstack([depot.reshape(1, 2), np.array(pts)])

data = {}
for n in [50, 100, 200]:
    instances = [_gen(n, seed=42+i) for i in range(3)]
    hms = []; tours = []
    for pts in instances:
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
            ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
            for i in range(50):
                t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
                xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
            hm = xt.float().cpu().numpy().squeeze() + 1e-6
        np_pts = pts.astype(np.float64)
        t, _ = merge_tours(hm[np.newaxis,:,:], np_pts, None, sparse_graph=False, parallel_sampling=1)
        s, _ = batched_two_opt_torch(np_pts, np.array(t).astype('int64'), max_iterations=1000, device=device)
        hms.append(hm); tours.append(s[0].tolist())
    data[str(n)] = {'pts': [p.tolist() for p in instances], 'hm': [h.tolist() for h in hms], 'dif_tours': tours}
    print('TSP-%d done' % n)

with open(os.path.join(_p, 'outputs', '_clustered_data.pkl'), 'wb') as f:
    pickle.dump(data, f)
print('All heatmaps saved')
'''

subprocess.run([_venv_py, '-c', gen_code], check=True, timeout=600)
print('Done.\n')

# ---- Step 2: Run all methods ----
with open(os.path.join(_project, 'outputs', '_clustered_data.pkl'), 'rb') as f:
    data = pickle.load(f)

# DualOpt setup
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

# LKH setup
import lkh, tsplib95

all_results = {}

for size_str in ['50', '100', '200']:
    n = int(size_str)
    d = data[size_str]
    print('\n' + '=' * 50)
    print('CLUSTERED TSP-%d (3 instances)' % n)
    print('=' * 50)

    results = {m: [] for m in ['NN','Christofides','C+2opt','DIFUSCO','DualOpt','LKH3']}

    for idx in range(3):
        pts = np.array(d['pts'][idx], dtype=np.float64)
        dist_mat = compute_distance_matrix(pts)
        print('\n[Instance %d]' % (idx+1))

        # NN
        t0 = time.time()
        tour, _ = nearest_neighbor_tsp(pts)
        results['NN'].append({'cost': tour_cost(dist_mat, tour), 'time': time.time()-t0})

        # Christofides
        t0 = time.time()
        tour, _ = christofides_tsp(pts)
        results['Christofides'].append({'cost': tour_cost(dist_mat, tour), 'time': time.time()-t0})

        # C+2opt
        t0 = time.time()
        tour_c2opt, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
        results['C+2opt'].append({'cost': tour_cost(dist_mat, tour_c2opt), 'time': time.time()-t0})

        # DIFUSCO+2opt (pre-computed in subprocess)
        dif_tour = d['dif_tours'][idx]
        results['DIFUSCO'].append({'cost': tour_cost(dist_mat, dif_tour), 'time': 0})  # time in subprocess

        # DualOpt
        t0 = time.time()
        it = tour_c2opt[:-1] if tour_c2opt[-1]==tour_c2opt[0] else tour_c2opt
        seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
        for rid in range(3):
            rlen = [50,20,10][rid]
            if rlen < seeds.shape[1]:
                seeds = LCP_TSP(seeds, gc, revisers[rid], rlen, [25,10,5][rid])
        cost_dual = (seeds[:,1:]-seeds[:,:-1]).norm(p=2,dim=2).sum(1)+(seeds[:,0]-seeds[:,-1]).norm(p=2,dim=1)
        results['DualOpt'].append({'cost': cost_dual.item(), 'time': time.time()-t0})

        # LKH3
        t0 = time.time()
        problem = tsplib95.models.StandardProblem()
        problem.name='TSP'; problem.type='TSP'; problem.dimension=n
        problem.edge_weight_type='EUC_2D'
        problem.node_coords={i+1:(float(pts[i][0]*1e6),float(pts[i][1]*1e6)) for i in range(n)}
        sol = lkh.solve('LKH.exe', problem=problem, max_trials=100, runs=5)
        tour = [r-1 for r in sol[0]]
        if tour[0]!=tour[-1]: tour.append(tour[0])
        results['LKH3'].append({'cost': tour_cost(dist_mat, tour), 'time': time.time()-t0})

    # Print summary for this size
    lkh_mean = np.mean([r['cost'] for r in results['LKH3']])
    for name in ['NN','Christofides','C+2opt','DIFUSCO','DualOpt','LKH3']:
        costs = [r['cost'] for r in results[name]]
        times = [r['time'] for r in results[name]]
        gap = (np.mean(costs)/lkh_mean - 1)*100
        print('  %-15s: %.4f +- %.4f | %.2fs | gap=%+.1f%%' % (
            name, np.mean(costs), np.std(costs), np.mean(times), gap))

    all_results[size_str] = {name: {'cost_mean': float(np.mean([r['cost'] for r in results[name]])),
                                     'cost_std': float(np.std([r['cost'] for r in results[name]])),
                                     'time_mean': float(np.mean([r['time'] for r in results[name]])),
                                     'gap_vs_lkh': float((np.mean([r['cost'] for r in results[name]])/lkh_mean-1)*100)}
                              for name in results}

# ---- Final Summary ----
print('\n\n' + '=' * 65)
print('CLUSTERED TSP BENCHMARK — Final Summary')
print('=' * 65)
print('Gap vs LKH3 (%):')
print('%s  %s  %s  %s' % ('Method'.ljust(18), 'TSP-50'.rjust(8), 'TSP-100'.rjust(8), 'TSP-200'.rjust(8)))
print('-' * 48)
for name in ['NN','Christofides','C+2opt','DIFUSCO','DualOpt','LKH3']:
    row = name.ljust(18)
    for s in ['50','100','200']:
        row += ('%+.1f%%' % all_results[s][name]['gap_vs_lkh']).rjust(8)
    print(row)

print('\nTime (seconds):')
print('%s  %s  %s  %s' % ('Method'.ljust(18), 'TSP-50'.rjust(8), 'TSP-100'.rjust(8), 'TSP-200'.rjust(8)))
print('-' * 48)
for name in ['NN','Christofides','C+2opt','DIFUSCO','DualOpt','LKH3']:
    row = name.ljust(18)
    for s in ['50','100','200']:
        t = all_results[s][name]['time_mean']
        row += ('%.1fs'%t).rjust(8) if t < 60 else ('%.0fs'%t).rjust(8)
    print(row)

with open(os.path.join(_project, 'outputs', 'clustered_benchmark.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# ---- Visualization ----
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for idx, size_str in enumerate(['50', '100', '200']):
    n = int(size_str)
    pts = np.array(data[size_str]['pts'][0], dtype=np.float64)
    dif_tour = data[size_str]['dif_tours'][0]

    # Get best tour from LKH
    problem = tsplib95.models.StandardProblem()
    problem.name='TSP'; problem.type='TSP'; problem.dimension=n
    problem.edge_weight_type='EUC_2D'
    problem.node_coords={i+1:(float(pts[i][0]*1e6),float(pts[i][1]*1e6)) for i in range(n)}
    sol = lkh.solve('LKH.exe', problem=problem, max_trials=100, runs=5)
    lkh_tour = [r-1 for r in sol[0]]
    if lkh_tour[0]!=lkh_tour[-1]: lkh_tour.append(lkh_tour[0])

    tour_c2opt_viz, _ = christofides_with_2opt(pts, max_2opt_iterations=200)

    for col, (name, tour) in enumerate([('C+2opt', tour_c2opt_viz), ('DIFUSCO+2opt', dif_tour), ('LKH3', lkh_tour)]):
        ax = axes[1 if idx > 0 else 0, col]
        ax.scatter(pts[1:,0], pts[1:,1], c='steelblue', s=30, edgecolors='black', linewidth=0.3, zorder=3)
        ax.scatter(pts[0,0], pts[0,1], c='red', s=100, marker='*', edgecolors='darkred', zorder=10)
        for i in range(len(tour)-1):
            ax.plot([pts[tour[i],0], pts[tour[i+1],0]],
                    [pts[tour[i],1], pts[tour[i+1],1]], '-', color='darkblue', linewidth=0.8, alpha=0.6)
        c = tour_cost(compute_distance_matrix(pts), tour) if name != 'LKH3' else \
            tour_cost(compute_distance_matrix(pts), tour)
        ax.set_title('TSP-%d: %s\ncost=%.4f' % (n, name, c), fontsize=10, fontweight='bold')
        ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
        ax.set_aspect('equal'); ax.axis('off')

plt.suptitle('Clustered Delivery TSP — A Dataset Neither DIFUSCO Nor DualOpt Tested\n'
             'C+2opt vs DIFUSCO+2opt vs LKH3 on Realistic Neighborhood Topology',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
outpath = os.path.join(_project, 'outputs', 'clustered_benchmark.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved:', outpath)
