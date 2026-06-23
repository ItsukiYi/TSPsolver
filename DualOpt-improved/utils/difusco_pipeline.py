r"""Improvement #2: DIFUSCO -> DualOpt Pipeline.

Replaces DualOpt's first step with DIFUSCO diffusion output:

  1. DIFUSCO: 50-step denoising -> edge heatmap
  2. Greedy merge: heatmap -> valid TSP tour
  3. DualOpt revisers: k=50 -> k=20 -> k=10 refinement
"""

import os, sys, time
import numpy as np
import torch


def run_difusco_dualopt_pipeline(points, dualopt_path, difusco_ckpt_path,
                                  two_opt_iterations=0, verbose=False):
    _project = os.path.dirname(os.path.abspath(dualopt_path))
    n = len(points)

    # ---- Phase 1: DIFUSCO ----
    _saved_path = list(sys.path)

    difusco_pkg = os.path.join(_project, 'DIFUSCO-main', 'difusco')
    difusco_root = os.path.join(_project, 'DIFUSCO-main')
    sys.path = [difusco_pkg, difusco_root] + [p for p in sys.path
        if 'difusco' not in p.lower() and 'DualOpt' not in p]

    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'):
            del sys.modules[k]

    from pl_tsp_model import TSPModel
    from argparse import Namespace
    from utils.tsp_utils import merge_tours, batched_two_opt_torch
    from utils.diffusion_schedulers import InferenceSchedule

    dif_args = Namespace(
        diffusion_type='categorical', diffusion_schedule='cosine',
        diffusion_steps=1000, inference_diffusion_steps=50,
        inference_schedule='cosine', inference_trick='ddim',
        n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation='sum',
        two_opt_iterations=two_opt_iterations,
        parallel_sampling=1, sequential_sampling=1,
        save_numpy_heatmap=False, storage_path='.',
        training_split='data/tsp_problems/tsp50_test.txt',
        validation_split='data/tsp_problems/tsp50_test.txt',
        test_split='data/tsp_problems/tsp50_test.txt',
        batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
        lr_scheduler='cosine-decay', num_epochs=50, num_workers=0,
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name='pipeline')

    device = torch.device('cuda')
    model = TSPModel.load_from_checkpoint(difusco_ckpt_path, param_args=dif_args, strict=False)
    model = model.to(device); model.eval()

    t0 = time.time()
    with torch.no_grad():
        pts_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1_i, t2_i = ts(i); t1_i=np.array([t1_i]).astype(int); t2_i=np.array([t2_i]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1_i, device, None, target_t=t2_i)
        heatmap = xt.float().cpu().numpy().squeeze() + 1e-6

    np_pts = points.astype(np.float64)
    tours, merge_iters = merge_tours(heatmap[np.newaxis,:,:], np_pts, None,
                                      sparse_graph=False, parallel_sampling=1)
    if two_opt_iterations > 0:
        solved, _ = batched_two_opt_torch(np_pts, np.array(tours).astype('int64'),
                                           max_iterations=two_opt_iterations, device=device)
        init_tour = solved[0].tolist()
    else:
        init_tour = tours[0]

    dif_time = time.time() - t0
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(points))
    dif_cost = sum(dist_mat[init_tour[i], init_tour[i+1]] for i in range(len(init_tour)-1))

    if verbose:
        print(f'  [DIFUSCO] cost={dif_cost:.4f} ({dif_time:.1f}s)')

    # ---- Phase 2: DualOpt Reviser ----
    sys.path[:] = _saved_path
    sys.path.insert(0, dualopt_path)

    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'):
            del sys.modules[k]

    from utils import load_model
    from utils.functions import second_step, load_problem

    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(dualopt_path, 'pretrained', f'local_{size}', 'epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
    opts = O()

    if init_tour[0] == init_tour[-1]:
        init_tour = init_tour[:-1]
    seeds = torch.from_numpy(points[init_tour]).float().unsqueeze(0).to(device)
    get_cost = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

    t2 = time.time()
    _, cost_revised = second_step(seeds, get_cost, opts, revisers)
    dual_time = time.time() - t2
    dual_cost = cost_revised.min().item()

    if verbose:
        impr = (dif_cost - dual_cost) / dif_cost * 100
        print(f'  [DualOpt] cost={dual_cost:.4f} ({dual_time:.1f}s, {impr:+.1f}%)')

    return {
        'difusco_cost': float(dif_cost), 'dualopt_cost': float(dual_cost),
        'difusco_time': dif_time, 'dualopt_time': dual_time,
        'total_time': dif_time + dual_time,
        'improvement_pct': float((dif_cost - dual_cost) / dif_cost * 100),
    }
