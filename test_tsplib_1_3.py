"""Test Improvements #1 and #3 on TSPLIB."""

import sys, os, time, numpy as np, torch

_project = os.path.dirname(__file__)
sys.path.append(os.path.join(_project, 'src'))
from src.tsplib_loader import load_tsplib_instance
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

instances = ['eil51', 'berlin52', 'eil76', 'kroA100']
device = torch.device('cuda')

print(f'{"=" * 60}')
print(f'TSPLIB: Improvements #1 (Heatmap) & #3 (Adaptive)')
print(f'{"=" * 60}')

for name in instances:
    pts, _, opt, display = load_tsplib_instance(name)
    n = len(pts)
    print(f'\n--- {name} (n={n}, opt={opt}) ---')

    # ---- DIFUSCO heatmap ----
    sys.path = [os.path.join(_project, 'DIFUSCO-main', 'difusco'),
                os.path.join(_project, 'DIFUSCO-main')] + \
               [p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
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

    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        heatmap = xt.float().cpu().numpy().squeeze() + 1e-6

    # ---- C+2opt initial tour ----
    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    c2_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
    it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
    print(f'  C+2opt init: {c2_cost:.2f} (gap={c2_cost/opt-1:.2%})' if opt else f'  C+2opt: {c2_cost:.2f}')

    # ---- Original DualOpt ----
    sys.path = [os.path.join(_project, 'DualOpt-main')] + \
               [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
    sys.path.append(os.path.join(_project, 'src'))
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]

    from utils import load_model
    from utils.functions import second_step, load_problem

    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(_project, f'DualOpt-main/pretrained/local_{size}/epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
    opts = O()
    gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

    _, cost_orig = second_step(seeds.clone(), gc, opts, revisers)
    gap_orig = cost_orig.min().item()/opt-1 if opt else 0
    print(f'  Original: {cost_orig.min().item():.2f} (gap={gap_orig:.2%})')

    # ---- #1 Heatmap-Guided (k=20,10 only) ----
    sys.path = [os.path.join(_project, 'DualOpt-improved')] + \
               [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
    sys.path.append(os.path.join(_project, 'src'))
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]

    from utils.heatmap_guide import heatmap_guided_LCP_TSP as hg
    from utils import load_model as lm2
    from utils.functions import LCP_TSP

    revisers_i = []
    for size in [50, 20, 10]:
        path = os.path.join(_project, f'DualOpt-improved/pretrained/local_{size}/epoch-100.pt')
        r, _ = lm2(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        revisers_i.append(r)

    seeds1 = seeds.clone()
    seeds1 = LCP_TSP(seeds1, gc, revisers_i[0], 50, 25)  # k=50: original
    seeds1 = hg(seeds1, gc, revisers_i[1], 20, 10, heatmap, tour_perm=it)  # k=20: heatmap-guided
    seeds1 = hg(seeds1, gc, revisers_i[2], 10, 5, heatmap, tour_perm=it)   # k=10: heatmap-guided
    cost1 = (seeds1[:,1:]-seeds1[:,:-1]).norm(p=2,dim=2).sum(1)+(seeds1[:,0]-seeds1[:,-1]).norm(p=2,dim=1)
    gap1 = cost1.min().item()/opt-1 if opt else 0
    i1 = (cost_orig.min().item()-cost1.min().item())/cost_orig.min().item()*100
    print(f'  #1 Heatmap: {cost1.min().item():.2f} (gap={gap1:.2%}, vs orig {i1:+.1f}%)')

    # ---- #3 Adaptive Window ----
    from utils.adaptive_reviser import adaptive_window_LCP_TSP as aw

    seeds3 = seeds.clone()
    tour_np = np.array(it)
    for rid in range(len(revisers_i)):
        rlen = opts.revision_lens[rid]
        if rlen < seeds3.shape[1]:
            seeds3 = aw(seeds3, gc, revisers_i[rid], rlen, opts.revision_iters[rid],
                        pts, tour_np)
        else:
            seeds3 = LCP_TSP(seeds3, gc, revisers_i[rid], rlen, opts.revision_iters[rid])
    cost3 = (seeds3[:,1:]-seeds3[:,:-1]).norm(p=2,dim=2).sum(1)+(seeds3[:,0]-seeds3[:,-1]).norm(p=2,dim=1)
    gap3 = cost3.min().item()/opt-1 if opt else 0
    i3 = (cost_orig.min().item()-cost3.min().item())/cost_orig.min().item()*100
    print(f'  #3 Adaptive: {cost3.min().item():.2f} (gap={gap3:.2%}, vs orig {i3:+.1f}%)')

# Summary
print(f'\n{"=" * 60}')
print(f'SUMMARY: Gap vs Optimal (%)')
print(f'{"=" * 60}')
print(f'{"Instance":<12s} {"Orig":>8s} {"#1 Heatmap":>10s} {"#3 Adaptive":>10s}')
print(f'{"-" * 45}')
# (results collected above, just print the summary table)

print(f'\nDone! See individual results above.')
