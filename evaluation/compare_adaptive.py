"""Improvement #3: Adaptive Window Sizing comparison."""

import sys, os, argparse, time, json, math
import numpy as np
import torch

_project = os.path.dirname(__file__)
sys.path = [p for p in sys.path if 'DualOpt' not in p and 'src' not in p and 'difusco' not in p.lower()]
sys.path.insert(0, os.path.join(_project, 'DualOpt-improved'))
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import second_step, load_problem, LCP_TSP
from utils.adaptive_reviser import adaptive_window_LCP_TSP
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost


def load_data(num_test=10):
    instances = []
    with open(os.path.join(_project, 'data/tsp_problems/tsp50_test.txt')) as f:
        for i, line in enumerate(f):
            if i >= num_test: break
            line = line.strip()
            if not line: continue
            parts = line.split(' output ')
            coords = [float(x) for x in parts[0].split()]
            pts = np.array([[coords[j], coords[j+1]] for j in range(0, len(coords), 2)])
            tour = [int(t)-1 for t in parts[1].split()]
            instances.append((pts, tour))
    return instances


def run_adaptive(pts, init_tour):
    """Run DualOpt with adaptive window sizing."""
    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(_project, f'DualOpt-improved/pretrained/local_{size}/epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to('cuda'); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
    opts = O()
    get_cost = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

    if init_tour[-1] == init_tour[0]:
        init_tour = init_tour[:-1]
    seeds = torch.from_numpy(pts[init_tour]).float().unsqueeze(0).to('cuda')
    tour_np = np.array(init_tour)

    for rid in range(len(revisers)):
        rlen = opts.revision_lens[rid]
        if rlen < seeds.shape[1]:
            # Adaptive: use 2-opt diagnostic for k=20 and k=10
            seeds = adaptive_window_LCP_TSP(
                seeds, get_cost, revisers[rid], rlen,
                opts.revision_iters[rid], pts, tour_np
            )
        else:
            # k=50 covers full tour, use standard reviser
            seeds = LCP_TSP(seeds, get_cost, revisers[rid], rlen,
                           opts.revision_iters[rid])

    cost_revised = (seeds[:,1:]-seeds[:,:-1]).norm(p=2,dim=2).sum(1) + \
                   (seeds[:,0]-seeds[:,-1]).norm(p=2,dim=1)
    return cost_revised.min().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-test', type=int, default=10)
    args = parser.parse_args()

    test_data = load_data(args.num_test)
    test_pts = [p for p, _ in test_data]

    print(f'{"=" * 60}')
    print(f'Improvement #3: Adaptive Window Sizing (TSP-50)')
    print(f'{"=" * 60}')

    orig_costs = []; adap_costs = []; c2opt_costs = []
    for idx, pts in enumerate(test_pts):
        print(f'\r[{idx+1}/{len(test_pts)}]', end='', flush=True)

        # Initial tour (same for both)
        init_tour, _ = christofides_with_2opt(pts, max_2opt_iterations=100)

        # Original DualOpt
        t0 = time.time()
        # (re-use the standard evaluate flow)
        from utils.functions import second_step as ss_orig
        revisers_o = []
        for size in [50, 20, 10]:
            path = os.path.join(_project, f'DualOpt-main/pretrained/local_{size}/epoch-100.pt')
            r, _ = load_model(path, is_local=True)
            r.to('cuda'); r.eval(); r.set_decode_type('greedy')
            revisers_o.append(r)
        class O: revision_lens=[50,20,10]; revision_iters=[25,10,5]; problem='tsp'; lkh_layer_number=2
        opts_o = O()
        gc_o = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
        it = init_tour[:-1] if init_tour[-1]==init_tour[0] else init_tour
        seeds_o = torch.from_numpy(pts[it]).float().unsqueeze(0).to('cuda')
        _, cost_o = ss_orig(seeds_o, gc_o, opts_o, revisers_o)
        orig_costs.append(cost_o.min().item())

        # Adaptive
        try:
            cost_a = run_adaptive(pts, init_tour)
        except Exception as e:
            cost_a = float('nan')
        adap_costs.append(cost_a)

        # C+2opt baseline
        tour_c, _ = christofides_with_2opt(pts, max_2opt_iterations=1000)
        c2opt_costs.append(tour_cost(compute_distance_matrix(pts), tour_c))

    print(f'\n\n{"=" * 60}')
    print('RESULTS')
    print(f'{"=" * 60}')
    valid_a = [c for c in adap_costs if not math.isnan(c)]
    print(f'  Original DualOpt:    {np.mean(orig_costs):.4f} +- {np.std(orig_costs):.4f}')
    if valid_a:
        print(f'  Adaptive DualOpt:    {np.mean(valid_a):.4f} +- {np.std(valid_a):.4f}')
        delta = (np.mean(valid_a)-np.mean(orig_costs))/np.mean(orig_costs)*100
        print(f'  Delta:                {delta:+.2f}%')
    print(f'  C+2opt baseline:     {np.mean(c2opt_costs):.4f} +- {np.std(c2opt_costs):.4f}')

    # Per-instance
    if valid_a:
        print(f'\n  Per-instance:')
        for i in range(len(test_pts)):
            if not math.isnan(adap_costs[i]):
                d = (adap_costs[i]-orig_costs[i])/orig_costs[i]*100
                m = '++' if d < -0.5 else ('+' if d < 0 else ('-' if d < 0.5 else '--'))
                print(f'    {i:2d}: orig={orig_costs[i]:.4f} adap={adap_costs[i]:.4f} ({d:+.2f}%) {m}')


if __name__ == '__main__':
    main()
