"""Test speed vs quality tradeoff for Improvements #1 and #3."""

import sys, os, time, numpy as np, torch

_project = os.path.dirname(__file__)

# Generate TSP-100 instances
np.random.seed(42)
instances = [np.random.rand(100, 2) for _ in range(3)]

# Setup paths
sys.path = [os.path.join(_project, 'DualOpt-improved'),
            os.path.join(_project, 'DualOpt-main')] + \
           [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import second_step, load_problem, LCP_TSP
from utils.heatmap_guide import heatmap_guided_LCP_TSP as hg
from src.algorithms import christofides_with_2opt

# Pre-generate heatmaps
print('Generating DIFUSCO heatmaps...')
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
for k in list(sys.modules.keys()):
    if k.startswith('utils'): del sys.modules[k]

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

heatmaps = []
for pts in instances:
    n = len(pts)
    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        heatmaps.append(xt.float().cpu().numpy().squeeze() + 1e-6)

print('Done.')

# Restore DualOpt paths
sys.path = [os.path.join(_project, 'DualOpt-improved'),
            os.path.join(_project, 'DualOpt-main')] + \
           [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
sys.path.append(os.path.join(_project, 'src'))
for k in list(sys.modules.keys()):
    if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]

from utils import load_model
from utils.functions import second_step, load_problem, LCP_TSP
from utils.heatmap_guide import heatmap_guided_LCP_TSP as hg
from src.algorithms import christofides_with_2opt

# Load reviser models
revisers_orig = []; revisers_impr = []
for dualopt in ['DualOpt-main', 'DualOpt-improved']:
    store = revisers_orig if 'main' in dualopt else revisers_impr
    for size in [50, 20, 10]:
        path = os.path.join(_project, dualopt, 'pretrained', f'local_{size}', 'epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        store.append(r)

class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
opts = O()
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

# ---- Compare all methods ----
methods = {
    'Original DualOpt': lambda pts, it, hm, idx: None,  # handled below
    'Heatmap-Guided (k=20,10)': lambda pts, it, hm, idx: None,
    'Heatmap-Guided (all k)': lambda pts, it, hm, idx: None,
}

print(f'\n{"=" * 65}')
print(f'SPEED vs QUALITY COMPARISON (TSP-100, 3 instances)')
print(f'{"=" * 65}')
print(f'{"Method":<30s} {"Cost":>8s} {"Time":>8s} {"vs Orig":>8s} {"Speedup":>8s}')
print(f'{"-" * 65}')

orig_times = []; orig_costs = []
for idx, pts in enumerate(instances):
    it, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    if it[-1] == it[0]: it = it[:-1]
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)

    t0 = time.time()
    _, cost = second_step(seeds.clone(), gc, opts, revisers_orig)
    t = time.time() - t0
    orig_times.append(t); orig_costs.append(cost.min().item())

print(f'  {"Original DualOpt":<30s} {np.mean(orig_costs):8.4f} {np.mean(orig_times):7.3f}s {"-":>8s} {"1.00x":>8s}')

# Heatmap-guided (k=20,10 only — keep k=50 original)
hg_times = []; hg_costs = []
for idx, pts in enumerate(instances):
    it, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    if it[-1] == it[0]: it = it[:-1]
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)

    t0 = time.time()
    # k=50: original
    seeds = LCP_TSP(seeds, gc, revisers_impr[0], 50, 25)
    # k=20, k=10: heatmap-guided
    seeds = hg(seeds, gc, revisers_impr[1], 20, 10, heatmaps[idx], tour_perm=it)
    seeds = hg(seeds, gc, revisers_impr[2], 10, 5, heatmaps[idx], tour_perm=it)
    cost = (seeds[:,1:]-seeds[:,:-1]).norm(p=2,dim=2).sum(1) + \
           (seeds[:,0]-seeds[:,-1]).norm(p=2,dim=1)
    t = time.time() - t0
    hg_times.append(t); hg_costs.append(cost.min().item())

delta = (np.mean(hg_costs)-np.mean(orig_costs))/np.mean(orig_costs)*100
speedup = np.mean(orig_times)/np.mean(hg_times)
print(f'  {"Heatmap-Guided":<30s} {np.mean(hg_costs):8.4f} {np.mean(hg_times):7.3f}s {delta:+7.2f}% {speedup:7.2f}x')

# ---- Also: what if we skip reviser ENTIRELY on k=10 (just 2 passes)?
light_times = []; light_costs = []
for idx, pts in enumerate(instances):
    it, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    if it[-1] == it[0]: it = it[:-1]
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)

    t0 = time.time()
    seeds = LCP_TSP(seeds, gc, revisers_orig[0], 50, 25)
    seeds = LCP_TSP(seeds, gc, revisers_orig[1], 20, 10)
    # Skip k=10 entirely
    cost = (seeds[:,1:]-seeds[:,:-1]).norm(p=2,dim=2).sum(1) + \
           (seeds[:,0]-seeds[:,-1]).norm(p=2,dim=1)
    t = time.time() - t0
    light_times.append(t); light_costs.append(cost.min().item())

delta_l = (np.mean(light_costs)-np.mean(orig_costs))/np.mean(orig_costs)*100
speedup_l = np.mean(orig_times)/np.mean(light_times)
print(f'  {"2 revisers only (no k=10)":<30s} {np.mean(light_costs):8.4f} {np.mean(light_times):7.3f}s {delta_l:+7.2f}% {speedup_l:7.2f}x')

print(f'\n  SUMMARY:')
print(f'    Original DualOpt:   cost={np.mean(orig_costs):.4f}  time={np.mean(orig_times):.3f}s')
print(f'    Heatmap-Guided:     cost={np.mean(hg_costs):.4f}  time={np.mean(hg_times):.3f}s  speedup={speedup:.2f}x  delta={delta:+.2f}%')
print(f'    2 revisers only:    cost={np.mean(light_costs):.4f}  time={np.mean(light_times):.3f}s  speedup={speedup_l:.2f}x  delta={delta_l:+.2f}%')
