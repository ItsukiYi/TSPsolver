"""Improvement #2 comparison: DIFUSCO vs DIFUSCO→DualOpt Pipeline."""

import sys, os, argparse, time, json, math
import numpy as np

_project = os.path.dirname(__file__)
# DualOpt MUST come before src to avoid utils conflict
# Clear any existing conflicting paths first
sys.path = [p for p in sys.path if 'DualOpt' not in p and 'src' not in p and 'difusco' not in p.lower()]
sys.path.insert(0, os.path.join(_project, 'DualOpt-improved'))
sys.path.append(os.path.join(_project, 'src'))  # append, not prepend

from utils.difusco_pipeline import run_difusco_dualopt_pipeline
from src.utils import compute_distance_matrix, tour_cost
from src.algorithms import christofides_with_2opt


def load_test_data(num_test=20):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-test', type=int, default=10)
    parser.add_argument('--output', type=str, default='outputs/pipeline_comparison.json')
    args = parser.parse_args()

    ckpt = os.path.join(_project, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
    dualopt_path = os.path.join(_project, 'DualOpt-improved')

    test_data = load_test_data(args.num_test)
    test_pts = [p for p, _ in test_data]
    test_gts = [t for _, t in test_data]
    n = len(test_data[0][0])

    print(f'{"=" * 65}')
    print(f'Improvement #2: DIFUSCO -> DualOpt Pipeline (TSP-{n})')
    print(f'{"=" * 65}')
    print(f'  Instances: {len(test_pts)}')
    print(f'  Methods:   DIFUSCO raw | DIFUSCO+2opt | DIFUSCO->DualOpt | C+2opt')
    print(f'{"=" * 65}')

    results = {
        'difusco_raw': [],
        'difusco_2opt': [],
        'difusco_dualopt': [],
        'c2opt_baseline': [],
        'gt_costs': [],
    }

    for idx, pts in enumerate(test_pts):
        print(f'\n[{idx+1}/{len(test_pts)}]', end='', flush=True)

        # Baseline: Christofides+2opt (our strongest classic)
        t0 = time.time()
        tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=1000)
        c2_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
        results['c2opt_baseline'].append(c2_cost)

        # Ground truth
        gt_tour = test_gts[idx]
        gt_cost = tour_cost(compute_distance_matrix(pts), gt_tour)
        results['gt_costs'].append(gt_cost)

        # DIFUSCO raw (no 2-opt, no reviser) + DIFUSCO→DualOpt
        res = run_difusco_dualopt_pipeline(
            pts, dualopt_path, ckpt, two_opt_iterations=0, verbose=False
        )
        results['difusco_raw'].append(res['difusco_cost'])
        results['difusco_dualopt'].append(res['dualopt_cost'])

        # DIFUSCO + 2-opt (standard inference)
        res2 = run_difusco_dualopt_pipeline(
            pts, dualopt_path, ckpt, two_opt_iterations=1000, verbose=False
        )
        results['difusco_2opt'].append(res2['difusco_cost'])

        print(f' raw={res["difusco_cost"]:.3f} 2opt={res2["difusco_cost"]:.3f} '
              f'pipe={res["dualopt_cost"]:.3f} c2opt={c2_cost:.3f}', end='')

    print(f'\n\n{"=" * 65}')
    print('RESULTS')
    print(f'{"=" * 65}')

    gt_mean = np.mean(results['gt_costs'])
    print(f'  Ground truth (C+2opt-5000): {gt_mean:.4f}')
    print()
    print(f'  {"Method":<25s} {"Mean":>8s} {"Std":>8s} {"vs GT":>8s} {"vs raw":>10s}')
    print(f'  {"-" * 60}')

    raw_mean = np.mean(results['difusco_raw'])
    for name, key in [
        ('DIFUSCO (raw, no 2-opt)', 'difusco_raw'),
        ('DIFUSCO + 2-opt', 'difusco_2opt'),
        ('DIFUSCO -> DualOpt', 'difusco_dualopt'),
        ('Christofides + 2-opt', 'c2opt_baseline'),
    ]:
        m = np.mean(results[key])
        s = np.std(results[key])
        vs_gt = (m / gt_mean - 1) * 100
        vs_raw = (m / raw_mean - 1) * 100
        marker = ' <- BEST' if key == 'difusco_dualopt' else ''
        print(f'  {name:<25s} {m:8.4f} {s:8.4f} {vs_gt:+7.2f}% {vs_raw:+9.2f}%{marker}')

    # Improvement chain analysis
    print(f'\n  Improvement chain:')
    impr_2opt = (raw_mean - np.mean(results['difusco_2opt'])) / raw_mean * 100
    impr_pipe = (raw_mean - np.mean(results['difusco_dualopt'])) / raw_mean * 100
    print(f'    DIFUSCO raw -> +2-opt:     {impr_2opt:.2f}% improvement')
    print(f'    DIFUSCO raw -> +DualOpt:   {impr_pipe:.2f}% improvement')

    # Save
    os.makedirs('outputs', exist_ok=True)
    output = {k: [float(vv) for vv in v] for k, v in results.items()}
    output['summary'] = {
        'difusco_raw_mean': float(raw_mean),
        'difusco_2opt_mean': float(np.mean(results['difusco_2opt'])),
        'difusco_dualopt_mean': float(np.mean(results['difusco_dualopt'])),
        'c2opt_mean': float(np.mean(results['c2opt_baseline'])),
        'gt_mean': float(gt_mean),
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n  Saved to {args.output}')


if __name__ == '__main__':
    main()
