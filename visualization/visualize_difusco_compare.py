"""DIFUSCO heatmap → tour comparison (standalone)."""
import sys, os, numpy as np, torch, argparse

_project = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
sys.path.insert(2, os.path.join(_project, 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pl_tsp_model import TSPModel
from argparse import Namespace
from utils.tsp_utils import merge_tours, batched_two_opt_torch
from utils.diffusion_schedulers import InferenceSchedule
from src.utils import compute_distance_matrix, tour_cost, generate_random_tsp_instance
from src.algorithms import nearest_neighbor_tsp

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=15)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

points = generate_random_tsp_instance(args.n, seed=args.seed)
n = len(points)
heatmap_cmap = LinearSegmentedColormap.from_list('heat', ['#FFFFFF', '#FFD700', '#FF8C00', '#FF0000', '#8B0000'])

cfg = Namespace(diffusion_type='categorical', diffusion_schedule='cosine',
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
model = TSPModel.load_from_checkpoint(ckpt, param_args=cfg, strict=False)
model = model.to(device); model.eval()

with torch.no_grad():
    pts_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
    xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
    ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
    for i in range(50):
        t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
        xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
    heatmap = xt.float().cpu().numpy().squeeze() + 1e-6

np_pts = points.astype(np.float64)
tours, _ = merge_tours(heatmap[np.newaxis, :, :], np_pts, None, sparse_graph=False, parallel_sampling=1)
solved, _ = batched_two_opt_torch(np_pts, np.array(tours).astype('int64'), max_iterations=1000, device=device)
diff_tour = solved[0].tolist()
diff_cost = tour_cost(compute_distance_matrix(points), diff_tour)

nn_tour, _ = nearest_neighbor_tsp(points)
nn_cost = tour_cost(compute_distance_matrix(points), nn_tour)

print(f'DIFUSCO cost={diff_cost:.3f}, NN cost={nn_cost:.3f}, improvement={(nn_cost-diff_cost)/nn_cost*100:.1f}%')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 1. Random noise (step 0)
ax = axes[0]
noise = np.random.rand(n, n)
ax.imshow(noise, cmap=heatmap_cmap, vmin=0, vmax=1, aspect='equal')
ax.set_title('Start: Random Noise', fontsize=12, fontweight='bold')
ax.set_xlabel('Node j'); ax.set_ylabel('Node i')

# 2. Final heatmap
ax = axes[1]
ax.imshow(heatmap, cmap=heatmap_cmap, vmin=0, vmax=1, aspect='equal')
ax.set_title('After 50 Denoising Steps\n(Edge Probabilities)', fontsize=12, fontweight='bold')
ax.set_xlabel('Node j'); ax.set_ylabel('Node i')

# 3. DIFUSCO tour
ax = axes[2]
ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=60, edgecolors='black')
for i in range(len(diff_tour) - 1):
    ax.plot([points[diff_tour[i], 0], points[diff_tour[i+1], 0]],
            [points[diff_tour[i], 1], points[diff_tour[i+1], 1]], '-', color='darkgreen', linewidth=2)
ax.scatter(points[0, 0], points[0, 1], c='gold', s=120, marker='*', edgecolors='black', zorder=5)
ax.set_title(f'DIFUSCO + 2-opt\ncost={diff_cost:.3f}', fontsize=12, fontweight='bold')
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

# 4. NN (baseline)
ax = axes[3]
ax.scatter(points[:, 0], points[:, 1], c='steelblue', s=60, edgecolors='black')
for i in range(len(nn_tour) - 1):
    ax.plot([points[nn_tour[i], 0], points[nn_tour[i+1], 0]],
            [points[nn_tour[i], 1], points[nn_tour[i+1], 1]], '-', color='crimson', linewidth=2)
ax.scatter(points[0, 0], points[0, 1], c='gold', s=120, marker='*', edgecolors='black', zorder=5)
ax.set_title(f'Nearest Neighbor\ncost={nn_cost:.3f}', fontsize=12, fontweight='bold')
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal'); ax.axis('off')

plt.suptitle('DIFUSCO Pipeline: Random Noise → Heatmap → Tour', fontsize=14, fontweight='bold')
plt.tight_layout()
out = 'outputs/visualizations/07_difusco_vs_nn.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
