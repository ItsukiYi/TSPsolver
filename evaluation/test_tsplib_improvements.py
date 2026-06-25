"""Test improvements #2 and #4 on TSPLIB compatible instances."""

import sys, os, time, numpy as np, torch

_project = os.path.dirname(__file__)
sys.path.append(os.path.join(_project, 'src'))
from src.tsplib_loader import load_tsplib_instance

instances = ['eil51', 'berlin52', 'eil76', 'kroA100']
device = torch.device('cuda')

results = {'orig': {}, 'pipeline': {}, 'freeze': {}}

for name in instances:
    pts, _, opt, display = load_tsplib_instance(name)
    n = len(pts)
    print(f'\n{"=" * 50}')
    print(f'{name} (n={n}, opt={opt})')
    print(f'{"=" * 50}')

    # ---- Get DIFUSCO heatmap ----
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

    t0 = time.time()
    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        heatmap = xt.float().cpu().numpy().squeeze() + 1e-6
    print(f'  DIFUSCO heatmap: {time.time()-t0:.1f}s')

    # ---- Get DIFUSCO initial tour (greedy merge) ----
    sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main'))
    from utils.tsp_utils import merge_tours
    np_pts = pts.astype(np.float64)
    tours, _ = merge_tours(heatmap[np.newaxis,:,:], np_pts, None, sparse_graph=False, parallel_sampling=1)
    dif_tour = tours[0]
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(pts))
    dif_cost = sum(dists[dif_tour[i], dif_tour[(i+1)%len(dif_tour)]] for i in range(len(dif_tour)))
    print(f'  DIFUSCO raw tour cost: {dif_cost:.2f} (gap={dif_cost/opt-1:.2%})' if opt else f'  DIFUSCO raw: {dif_cost:.2f}')

    # ---- Get C+2opt tour ----
    sys.path.append(os.path.join(_project, 'src'))
    from src.algorithms import christofides_with_2opt
    from src.utils import compute_distance_matrix, tour_cost
    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=200)
    c2_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
    print(f'  C+2opt tour cost: {c2_cost:.2f} (gap={c2_cost/opt-1:.2%})' if opt else f'  C+2opt: {c2_cost:.2f}')

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

    # Original: C+2opt initial -> DualOpt
    it_o = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
    seeds_o = torch.from_numpy(pts[it_o]).float().unsqueeze(0).to(device)
    _, cost_o = second_step(seeds_o, gc, opts, revisers)
    gap_o = cost_o.min().item()/opt-1 if opt else 0
    print(f'  Original DualOpt: {cost_o.min().item():.2f} (gap={gap_o:.2%})')

    # ---- #2 DIFUSCO -> DualOpt ----
    it_d = dif_tour[:-1] if dif_tour[-1]==dif_tour[0] else dif_tour
    seeds_d = torch.from_numpy(pts[it_d]).float().unsqueeze(0).to(device)
    _, cost_p = second_step(seeds_d, gc, opts, revisers)
    gap_p = cost_p.min().item()/opt-1 if opt else 0
    impr_p = (cost_o.min().item()-cost_p.min().item())/cost_o.min().item()*100
    print(f'  #2 Pipeline: {cost_p.min().item():.2f} (gap={gap_p:.2%}, vs orig {impr_p:+.1f}%)')
    results['pipeline'][name] = {'cost': cost_p.min().item(), 'gap': gap_p, 'impr': impr_p}

    # ---- #4 Fragment Freezing ----
    sys.path = [os.path.join(_project, 'DualOpt-improved')] + \
               [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
    sys.path.append(os.path.join(_project, 'src'))
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]

    from utils.freeze_reviser import compute_frozen_mask, freeze_guided_LCP_TSP
    from utils import load_model as lm2
    from utils.functions import LCP_TSP

    revisers2 = []
    for size in [50, 20, 10]:
        path = os.path.join(_project, f'DualOpt-improved/pretrained/local_{size}/epoch-100.pt')
        r, _ = lm2(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        revisers2.append(r)

    frozen_mask, agree_pct = compute_frozen_mask(pts, heatmap, tour_c2)

    seeds_f = torch.from_numpy(pts[it_o]).float().unsqueeze(0).to(device)
    for rid in range(len(revisers2)):
        rlen = opts.revision_lens[rid]
        if rlen < seeds_f.shape[1]:
            seeds_f = freeze_guided_LCP_TSP(seeds_f, gc, revisers2[rid], rlen,
                                             opts.revision_iters[rid], frozen_mask)
        else:
            seeds_f = LCP_TSP(seeds_f, gc, revisers2[rid], rlen, opts.revision_iters[rid])
    cost_f = (seeds_f[:,1:]-seeds_f[:,:-1]).norm(p=2,dim=2).sum(1)+(seeds_f[:,0]-seeds_f[:,-1]).norm(p=2,dim=1)
    gap_f = cost_f.min().item()/opt-1 if opt else 0
    impr_f = (cost_o.min().item()-cost_f.min().item())/cost_o.min().item()*100
    print(f'  #4 Freezing:  {cost_f.min().item():.2f} (gap={gap_f:.2%}, agree={agree_pct:.0f}%, vs orig {impr_f:+.1f}%)')
    results['freeze'][name] = {'cost': cost_f.min().item(), 'gap': gap_f, 'agree': agree_pct, 'impr': impr_f}
    results['orig'][name] = {'cost': cost_o.min().item(), 'gap': gap_o}

# Final summary
print(f'\n\n{"=" * 60}')
print(f'TSPLIB IMPROVEMENT SUMMARY')
print(f'{"=" * 60}')
print(f'{"Instance":<12s} {"Opt":>8s} {"C+2opt":>8s} {"Orig Dual":>9s} {"#2 Pipe":>8s} {"#4 Freeze":>9s}')
print(f'{"-" * 55}')
for name in instances:
    pts, _, opt, _ = load_tsplib_instance(name)
    c2_t, _ = christofides_with_2opt(pts, max_2opt_iterations=5000)
    c2_c = tour_cost(compute_distance_matrix(pts), c2_t)
    o = results['orig'].get(name, {})
    p = results['pipeline'].get(name, {})
    f = results['freeze'].get(name, {})
    print(f'{name:<12s} {opt:8.0f} {c2_c:8.1f} {o.get("cost",0):8.1f} {p.get("cost",0):8.1f} {f.get("cost",0):8.1f}')

print(f'\n  Gap vs Original DualOpt:')
print(f'  {"Instance":<12s} {"#2 Pipeline":>12s} {"#4 Freeze":>12s}')
for name in instances:
    p = results['pipeline'].get(name, {})
    f = results['freeze'].get(name, {})
    imp = p.get('impr', 0)
    imf = f.get('impr', 0)
    print(f'  {name:<12s} {imp:+11.1f}% {imf:+11.1f}%')
