"""Test Improvement #1 on TSP-100 — run via subprocesses to avoid path conflicts."""

import subprocess, sys, os, json, numpy as np

_project = os.path.dirname(__file__)
venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')

# Generate TSP-100 instances
np.random.seed(42)
instances = [np.random.rand(100, 2).tolist() for _ in range(5)]
np.save(os.path.join(_project, 'outputs', 'tsp100_test.npy'), np.array(instances))
data_path = os.path.join(_project, 'outputs', 'tsp100_test.npy')

# ---- Run Original DualOpt (separate process) ----
orig_script = """
import sys, os, json, numpy as np, torch
_project = r'{project}'
sys.path.insert(0, os.path.join(_project, 'DualOpt-main'))
sys.path.append(os.path.join(_project, 'src'))
from utils import load_model
from utils.functions import second_step, load_problem
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

instances = np.load(r'{data}').tolist()

revisers = []
for size in [50, 20, 10]:
    path = os.path.join(_project, f'DualOpt-main/pretrained/local_{{size}}/epoch-100.pt')
    r, _ = load_model(path, is_local=True)
    r.to('cuda'); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
opts = O()
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

results = []
for pts in instances:
    pts = np.array(pts)
    it, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    if it[-1] == it[0]: it = it[:-1]
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to('cuda')
    _, cost = second_step(seeds, gc, opts, revisers)
    results.append(cost.min().item())

print(json.dumps(results))
"""

guided_script = """
import sys, os, json, numpy as np, torch
_project = r'{project}'
# DIFUSCO first
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
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
ckpt = os.path.join(_project, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

instances = np.load(r'{data}').tolist()

# Get heatmaps for all instances first
heatmaps = []
for pts in instances:
    pts_np = np.array(pts); n = len(pts_np)
    with torch.no_grad():
        pts_t = torch.from_numpy(pts_np).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        heatmaps.append(xt.float().cpu().numpy().squeeze() + 1e-6)

print('HEATMAPS_READY')
sys.stdout.flush()

# Now run DualOpt-improved with heatmap guidance
import sys
# Clear DIFUSCO paths, add DualOpt-improved
sys.path = [os.path.join(_project, 'DualOpt-improved')] + [p for p in sys.path
    if 'difusco' not in p.lower() and 'DualOpt' not in p]
sys.path.append(os.path.join(_project, 'src'))

for k in list(sys.modules.keys()):
    if k.startswith('utils') or k.startswith('pl_'):
        del sys.modules[k]

from utils import load_model
from utils.functions import second_step, load_problem
from src.algorithms import christofides_with_2opt

revisers = []
for size in [50, 20, 10]:
    path = os.path.join(_project, f'DualOpt-improved/pretrained/local_{{size}}/epoch-100.pt')
    r, _ = load_model(path, is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
opts = O()
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

results = []
for idx, pts_list in enumerate(instances):
    pts = np.array(pts_list)
    it, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    if it[-1] == it[0]: it = it[:-1]
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to('cuda')
    _, cost = second_step(seeds, gc, opts, revisers, heatmap=heatmaps[idx], tour_perm=it)
    results.append(cost.min().item())

print(json.dumps(results))
"""

orig_fmt = orig_script.format(project=_project, data=data_path)
guided_fmt = guided_script.format(project=_project, data=data_path)

print('Running Original DualOpt...')
orig_out = subprocess.run([venv_py, '-c', orig_fmt], capture_output=True, text=True, timeout=600)
orig_costs = json.loads(orig_out.stdout.strip().split('\n')[-1])
print(f'  Original: {[f"{c:.4f}" for c in orig_costs]}')

print('Running Heatmap-Guided...')
guided_out = subprocess.run([venv_py, '-c', guided_fmt], capture_output=True, text=True, timeout=600)
# Find the JSON array in output (after HEATMAPS_READY marker)
lines = guided_out.stdout.strip().split('\n')
guided_costs = json.loads(lines[-1])
print(f'  Guided:   {[f"{c:.4f}" for c in guided_costs]}')

# Summary
print(f'\n{"=" * 60}')
print(f'RESULTS: TSP-100 (5 instances)')
print(f'{"=" * 60}')
print(f'  Original:  {np.mean(orig_costs):.4f}')
print(f'  Guided:    {np.mean(guided_costs):.4f}')
delta = (np.mean(guided_costs)-np.mean(orig_costs))/np.mean(orig_costs)*100
print(f'  Delta:     {delta:+.2f}%')
improved = sum(1 for o, g in zip(orig_costs, guided_costs) if g < o)
print(f'  Improved:  {improved}/5')
for i in range(5):
    d = (guided_costs[i]-orig_costs[i])/orig_costs[i]*100
    s = 'IMPROVED' if d < -0.01 else ('SAME' if abs(d)<0.01 else 'DEGRADED')
    print(f'    {i}: orig={orig_costs[i]:.4f} guided={guided_costs[i]:.4f} ({d:+.2f}%) [{s}]')
