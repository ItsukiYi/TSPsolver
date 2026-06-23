"""DIFUSCO ablation: inference steps + with/without 2-opt."""
import sys, os, numpy as np, torch, time, json

_project = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
sys.path.insert(2, os.path.join(_project, 'src'))

from pl_tsp_model import TSPModel
from argparse import Namespace
from utils.tsp_utils import batched_two_opt_torch, merge_tours
from utils.diffusion_schedulers import InferenceSchedule
from src.utils import compute_distance_matrix, tour_cost

instances = []
with open('data/tsp_problems/tsp50_test.txt') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        line = line.strip()
        if not line: continue
        parts = line.split(' output ')
        coords = [float(x) for x in parts[0].split()]
        pts = np.array([[coords[j], coords[j+1]] for j in range(0, len(coords), 2)])
        tour = [int(t)-1 for t in parts[1].split()]
        instances.append((pts, tour))

results = {}
device = torch.device('cuda')
ckpt = 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt'

configs = [
    (10, False, '10 steps, no 2-opt'),
    (10, True,  '10 steps + 2-opt'),
    (20, False, '20 steps, no 2-opt'),
    (20, True,  '20 steps + 2-opt'),
    (50, False, '50 steps, no 2-opt'),
    (50, True,  '50 steps + 2-opt'),
]

for infer_steps, use_2opt, label in configs:
    print(f'\n[{label}]')
    args = Namespace(diffusion_type='categorical', diffusion_schedule='cosine',
        diffusion_steps=1000, inference_diffusion_steps=infer_steps,
        inference_schedule='cosine', inference_trick='ddim',
        n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation='sum',
        two_opt_iterations=1000 if use_2opt else 0,
        parallel_sampling=1, sequential_sampling=1,
        save_numpy_heatmap=False, storage_path='.',
        training_split='data/tsp_problems/tsp50_test.txt',
        validation_split='data/tsp_problems/tsp50_test.txt',
        test_split='data/tsp_problems/tsp50_test.txt',
        batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
        lr_scheduler='cosine-decay', num_epochs=50, num_workers=0,
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name='ablation')

    model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
    model = model.to(device); model.eval()

    costs = []; times = []
    for idx, (pts, _) in enumerate(instances):
        print(f'\r  {idx+1}/{len(instances)}', end='', flush=True)
        n = len(pts)
        t0 = time.time()
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
            ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=infer_steps)
            for i in range(infer_steps):
                t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
                xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
            adj_mat = xt.float().cpu().numpy() + 1e-6
        np_pts = pts.astype(np.float64)
        tours, _ = merge_tours(adj_mat, np_pts, None, sparse_graph=False, parallel_sampling=1)
        if use_2opt:
            solved, _ = batched_two_opt_torch(np_pts, np.array(tours).astype('int64'),
                                               max_iterations=1000, device=device)
            tour = solved[0].tolist()
        else:
            tour = tours[0]
        cost = tour_cost(compute_distance_matrix(pts), tour)
        costs.append(cost)
        times.append(time.time() - t0)
    print()
    results[label] = {'mean_cost': float(np.mean(costs)), 'std_cost': float(np.std(costs)),
                      'mean_time': float(np.mean(times))}

gt_costs = [tour_cost(compute_distance_matrix(p), t) for p, t in instances]
gt_mean = float(np.mean(gt_costs))

print()
print('=' * 65)
print('DIFUSCO ABLATION RESULTS (10 TSP-50 instances)')
print('=' * 65)
print(f'{"Configuration":<30s} {"Mean":>8s} {"Std":>8s} {"vs GT":>8s} {"Time":>8s}')
print('-' * 62)
for label, r in results.items():
    gap = (r['mean_cost']/gt_mean - 1)*100
    print(f'{label:<30s} {r["mean_cost"]:8.4f} {r["std_cost"]:8.4f} {gap:+7.2f}% {r["mean_time"]:7.3f}s')
print(f'{"Ground truth (C+2opt-5000)":<30s} {gt_mean:8.4f}')
print()

with open('outputs/difusco_ablation.json', 'w') as f:
    json.dump({'results': results, 'gt_mean': gt_mean}, f, indent=2)
print('Saved to outputs/difusco_ablation.json')
