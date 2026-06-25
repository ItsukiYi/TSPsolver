r"""Confidence Calibration Study: How reliable are DIFUSCO heatmap scores?

Analyzes whether DIFUSCO's edge probabilities correlate with actual edge optimality.
Bin edges by confidence, measure the fraction that appear in the reference (LKH/C+2opt) tour.

Key insight: if DIFUSCO says P(edge)=0.9, does that edge actually appear 90% of the time?
"""

import sys, os, time, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_project = os.path.dirname(__file__)
sys.path.append(os.path.join(_project, 'src'))
from src.tsplib_loader import load_tsplib_instance, KNOWN_OPTIMAL_COSTS


def get_reference_edges(points, method='c2opt'):
    """Get reference (near-optimal) edge set."""
    from src.algorithms import christofides_with_2opt
    from src.utils import compute_distance_matrix, tour_cost

    if method == 'c2opt':
        tour, _ = christofides_with_2opt(points, max_2opt_iterations=5000)
    else:
        tour, _ = christofides_with_2opt(points, max_2opt_iterations=200)

    n = len(points)
    if tour[-1] == tour[0]:
        tour = tour[:-1]

    edges = set()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i+1) % len(tour)]
        edges.add((min(u, v), max(u, v)))
    return edges, tour


def run_calibration(pts, heatmap, ref_edges):
    """Bin edges by DIFUSCO confidence and compute hit rate per bin."""
    n = len(pts)
    bins = np.arange(0.0, 1.05, 0.05)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hits = np.zeros(len(bin_centers))
    counts = np.zeros(len(bin_centers))

    for i in range(n):
        for j in range(i+1, n):
            conf = heatmap[i, j]
            bin_idx = np.digitize(conf, bins) - 1
            if 0 <= bin_idx < len(bin_centers):
                counts[bin_idx] += 1
                if (i, j) in ref_edges:
                    hits[bin_idx] += 1

    rates = np.divide(hits, counts, out=np.zeros_like(hits), where=counts > 0)
    return bin_centers, rates, counts


# ---- Main ----
print('=' * 60)
print('Confidence Calibration Study')
print('=' * 60)

# Generate data: random TSP-50, TSP-100, TSP-200, TSPLIB instances
import subprocess
_venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')

# Generate heatmaps via subprocess
gen_code = r'''
import sys, os, pickle, numpy as np, torch, json
_p = r"''' + _project + r'''"
sys.path.insert(0, os.path.join(_p, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_p, 'DIFUSCO-main'))
from pl_tsp_model import TSPModel
from argparse import Namespace
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
    validation_examples=8, use_activation_checkpoint=False, fp16=False)
device = torch.device('cuda')
ckpt = os.path.join(_p, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

# Random instances
np.random.seed(42)
all_data = {}
for n in [50, 100, 200]:
    instances = [np.random.rand(n, 2).astype(np.float64) for _ in range(3)]
    hms = []
    for pts in instances:
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
            ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
            for i in range(50):
                t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
                xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
            hms.append(xt.float().cpu().numpy().squeeze() + 1e-6)
    all_data[str(n)] = {'pts': [p.tolist() for p in instances], 'hm': [h.tolist() for h in hms]}

# TSPLIB instances
sys.path.append(os.path.join(_p, 'src'))
from src.tsplib_loader import load_tsplib_instance
tsplib_names = ['eil51', 'berlin52', 'eil76', 'kroA100']
tsplib_data = {}
for name in tsplib_names:
    pts_raw, _, opt, _ = load_tsplib_instance(name)
    n = len(pts_raw)
    with torch.no_grad():
        pts_t = torch.from_numpy(pts_raw).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        hm = xt.float().cpu().numpy().squeeze() + 1e-6
    tsplib_data[name] = {'pts': pts_raw.tolist(), 'hm': hm.tolist(), 'opt': opt}

with open(os.path.join(_p, 'outputs', '_calib_data.pkl'), 'wb') as f:
    pickle.dump({'random': all_data, 'tsplib': tsplib_data}, f)
print('Heatmaps generated')
'''

print('Generating DIFUSCO heatmaps (subprocess)...')
subprocess.run([_venv_py, '-c', gen_code], check=True, timeout=600)
print('Done.\n')

# Analysis
with open(os.path.join(_project, 'outputs', '_calib_data.pkl'), 'rb') as f:
    data = pickle.load(f)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Panel 1: Random instances by size
ax = axes[0]
colors = {50: 'blue', 100: 'green', 200: 'red'}
for n_str, color in [('50', 'blue'), ('100', 'green'), ('200', 'red')]:
    all_rates = []
    all_counts = []
    n = int(n_str)
    for idx in range(3):
        pts = np.array(data['random'][n_str]['pts'][idx])
        hm = np.array(data['random'][n_str]['hm'][idx])
        ref_edges, _ = get_reference_edges(pts, 'c2opt')
        bins_c, rates, counts = run_calibration(pts, hm, ref_edges)
        all_rates.append(rates)
        all_counts.append(counts)

    mean_rates = np.mean(all_rates, axis=0)
    total_counts = np.sum(all_counts, axis=0)
    # Only plot bins with sufficient samples (>50)
    mask = total_counts > 50
    ax.plot(bins_c[mask], mean_rates[mask], '-o', color=color, linewidth=2,
            markersize=6, label='TSP-%d (random)' % n)

# Diagonal: perfect calibration
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
ax.set_xlabel('DIFUSCO Edge Confidence (binned)', fontsize=12)
ax.set_ylabel('Fraction of Edges in Optimal Tour', fontsize=12)
ax.set_title('Calibration: Random Uniform Instances', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Panel 2: TSPLIB vs Random
ax = axes[1]
# Random TSP-50 as reference
all_rates_50 = []
for idx in range(3):
    pts = np.array(data['random']['50']['pts'][idx])
    hm = np.array(data['random']['50']['hm'][idx])
    ref_edges, _ = get_reference_edges(pts, 'c2opt')
    bins_c, rates, counts = run_calibration(pts, hm, ref_edges)
    all_rates_50.append(rates)
mean_50 = np.mean(all_rates_50, axis=0)
mask_50 = np.ones_like(mean_50, dtype=bool)
ax.plot(bins_c, mean_50, '-o', color='blue', linewidth=2, markersize=6, label='TSP-50 random (in-dist)')

# TSPLIB instances
tsplib_colors = ['red', 'orange', 'purple', 'brown']
for idx, name in enumerate(['eil51', 'berlin52', 'eil76', 'kroA100']):
    pts = np.array(data['tsplib'][name]['pts'])
    hm = np.array(data['tsplib'][name]['hm'])
    ref_edges, _ = get_reference_edges(pts, 'c2opt')
    bins_c, rates, counts = run_calibration(pts, hm, ref_edges)
    mask = counts > 30
    ax.plot(bins_c[mask], rates[mask], '-s', color=tsplib_colors[idx], linewidth=2,
            markersize=5, label='%s (TSPLIB)' % name)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
ax.set_xlabel('DIFUSCO Edge Confidence (binned)', fontsize=12)
ax.set_ylabel('Fraction of Edges in Optimal Tour', fontsize=12)
ax.set_title('Calibration: TSPLIB vs In-Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.8)

plt.suptitle('DIFUSCO Heatmap Confidence Calibration\n(How well do edge probabilities predict optimality?)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
outpath = os.path.join(_project, 'outputs', 'calibration_study.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: %s' % outpath)

# Numerical summary
print('\n' + '=' * 60)
print('CALIBRATION SUMMARY')
print('=' * 60)
print('Higher is better. Values near diagonal = well-calibrated.')
print()
for label, pts_list, hm_list in [
    ('TSP-50 random', [np.array(data['random']['50']['pts'][i]) for i in range(3)],
     [np.array(data['random']['50']['hm'][i]) for i in range(3)]),
    ('TSP-100 random', [np.array(data['random']['100']['pts'][i]) for i in range(3)],
     [np.array(data['random']['100']['hm'][i]) for i in range(3)]),
    ('TSP-200 random', [np.array(data['random']['200']['pts'][i]) for i in range(3)],
     [np.array(data['random']['200']['hm'][i]) for i in range(3)]),
]:
    all_r = []
    for pts, hm in zip(pts_list, hm_list):
        ref, _ = get_reference_edges(pts, 'c2opt')
        _, rates, counts = run_calibration(pts, hm, ref)
        # Weighted average: high-confidence bins
        high_conf = rates[14:]  # bins 0.7-1.0
        high_counts = counts[14:]
        if high_counts.sum() > 0:
            avg_high = np.average(high_conf, weights=high_counts)
        else:
            avg_high = 0
        all_r.append(avg_high)
    print('  %s: high-conf (0.7-1.0) avg hit rate = %.1f%%' % (label, np.mean(all_r)*100))

for name in ['eil51', 'berlin52', 'eil76', 'kroA100']:
    pts = np.array(data['tsplib'][name]['pts'])
    hm = np.array(data['tsplib'][name]['hm'])
    ref, _ = get_reference_edges(pts, 'c2opt')
    _, rates, counts = run_calibration(pts, hm, ref)
    high_conf = rates[14:]
    high_counts = counts[14:]
    avg_high = np.average(high_conf, weights=high_counts) if high_counts.sum() > 0 else 0
    print('  %s (TSPLIB): high-conf (0.7-1.0) avg hit rate = %.1f%%' % (name, avg_high*100))
