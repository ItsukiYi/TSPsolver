"""Quick test: Hyper-Edge Repair with DualOpt reviser."""

import sys, os, numpy as np, torch

_project = os.path.dirname(__file__)
sys.path = [os.path.join(_project, 'DualOpt-improved'),
            os.path.join(_project, 'DualOpt-main')] + \
           [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import second_step, load_problem
from utils.hyper_repair import hyper_repair_with_dualopt
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

# Load test data
instances = []
with open(os.path.join(_project, 'data/tsp_problems/tsp50_test.txt')) as f:
    for i, line in enumerate(f):
        if i >= 5: break
        line = line.strip()
        if not line: continue
        parts = line.split(' output ')
        coords = [float(x) for x in parts[0].split()]
        pts = np.array([[coords[j], coords[j+1]] for j in range(0, len(coords), 2)])
        instances.append(pts)

# Load models once
device = torch.device('cuda')
revisers = []
for size in [50, 20, 10]:
    path = os.path.join(_project, f'DualOpt-improved/pretrained/local_{size}/epoch-100.pt')
    r, _ = load_model(path, is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
opts = O()

# DIFUSCO heatmap generator
def get_heatmap(pts):
    sys.path = [os.path.join(_project, 'DIFUSCO-main', 'difusco'),
                os.path.join(_project, 'DIFUSCO-main')] + \
               [p for p in sys.path if 'DualOpt' not in p and 'src' not in p]
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
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
    ckpt = os.path.join(_project, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
    model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
    model = model.to(device); model.eval()
    n = len(pts)
    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        hm = xt.float().cpu().numpy().squeeze() + 1e-6
    return hm

print('=' * 60)
print('Improvement #5b: Hyper-Edge Repair (5 instances)')
print('=' * 60)

for idx, pts in enumerate(instances):
    print(f'\n[{idx+1}/5]')
    hm = get_heatmap(pts)
    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=100)
    init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
    it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2

    # Try different K values
    for K in [3, 5]:
        try:
            repaired, cost, improved = hyper_repair_with_dualopt(
                pts, it, hm, K, revisers, opts
            )
            status = 'IMPROVED' if improved else 'SAME'
            delta = (init_cost - cost) / init_cost * 100
            print(f'  K={K}: {init_cost:.4f} -> {cost:.4f} ({delta:+.2f}%) [{status}]')
        except Exception as e:
            print(f'  K={K}: FAILED - {e}')
