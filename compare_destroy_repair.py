"""Improvement #5: Destroy-and-Repair comparison."""

import sys, os, time, json, math, numpy as np, torch

_project = os.path.dirname(__file__)
sys.path = [os.path.join(_project, 'DualOpt-improved'),
            os.path.join(_project, 'DualOpt-main')] + \
           [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import second_step, load_problem
from utils.destroy_repair import destroy_repair_cycle
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost


def load_data(num=10):
    instances = []
    with open(os.path.join(_project, 'data/tsp_problems/tsp50_test.txt')) as f:
        for i, line in enumerate(f):
            if i >= num: break
            line = line.strip()
            if not line: continue
            parts = line.split(' output ')
            coords = [float(x) for x in parts[0].split()]
            pts = np.array([[coords[j], coords[j+1]] for j in range(0, len(coords), 2)])
            instances.append(pts)
    return instances


def get_difusco_heatmap(pts):
    dif_pkg = os.path.join(_project, 'DIFUSCO-main', 'difusco')
    dif_root = os.path.join(_project, 'DIFUSCO-main')
    sys.path = [dif_pkg, dif_root] + [p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
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

    device = torch.device('cuda')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-test', type=int, default=10)
    args = parser.parse_args()

    test_pts = load_data(args.num_test)
    n_test = len(test_pts)
    n = len(test_pts[0])
    device = torch.device('cuda')

    print(f'{"=" * 65}')
    print(f'Improvement #5: Destroy-and-Repair (TSP-{n}, {n_test} instances)')
    print(f'{"=" * 65}')

    orig_costs = []; dr_costs = []; c2opt_costs = []
    for idx, pts in enumerate(test_pts):
        print(f'\n[{idx+1}/{n_test}]', end='', flush=True)

        hm = get_difusco_heatmap(pts)
        tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=100)
        c2opt_costs.append(tour_cost(compute_distance_matrix(pts), tour_c2))

        # Original DualOpt
        sys.path = [os.path.join(_project, 'DualOpt-main')] + \
                   [p for p in sys.path if 'DualOpt' not in p and 'difusco' not in p.lower()]
        sys.path.append(os.path.join(_project, 'src'))
        for k in list(sys.modules.keys()):
            if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]

        from utils import load_model as lm
        from utils.functions import second_step as ss

        revisers = []
        for size in [50, 20, 10]:
            path = os.path.join(_project, f'DualOpt-main/pretrained/local_{size}/epoch-100.pt')
            r, _ = lm(path, is_local=True)
            r.to(device); r.eval(); r.set_decode_type('greedy')
            revisers.append(r)

        class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
        opts = O()
        gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

        it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
        seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
        _, cost_orig = ss(seeds, gc, opts, revisers)
        orig_costs.append(cost_orig.min().item())

        # Destroy-and-Repair (with DualOpt polish)
        tour_dr, cost_dr, history = destroy_repair_cycle(
            pts, it, hm, K_values=[3, 5, 7], cycles=3,
            dualopt_revisers=revisers, dualopt_opts=opts, verbose=False
        )
        dr_costs.append(cost_dr)

        impr = (cost_orig.min().item() - cost_dr) / cost_orig.min().item() * 100
        print(f' orig={cost_orig.min().item():.4f} dr={cost_dr:.4f} ({impr:+.2f}%)', end='')

    print(f'\n\n{"=" * 65}')
    print('RESULTS')
    print(f'{"=" * 65}')
    print(f'  Original DualOpt:     {np.mean(orig_costs):.4f} +- {np.std(orig_costs):.4f}')
    print(f'  Destroy-and-Repair:   {np.mean(dr_costs):.4f} +- {np.std(dr_costs):.4f}')
    delta = (np.mean(dr_costs)-np.mean(orig_costs))/np.mean(orig_costs)*100
    print(f'  Delta:                {delta:+.2f}%')
    better = sum(1 for o, d in zip(orig_costs, dr_costs) if d < o)
    print(f'  Improved:             {better}/{n_test}')

    print(f'\n  Per-instance:')
    for i in range(n_test):
        d = (dr_costs[i]-orig_costs[i])/orig_costs[i]*100
        m = 'IMPROVED' if d < -0.01 else ('SAME' if abs(d)<0.01 else 'DEGRADED')
        print(f'    {i:2d}: orig={orig_costs[i]:.4f} dr={dr_costs[i]:.4f} ({d:+.2f}%) [{m}]')


if __name__ == '__main__':
    import argparse
    main()
