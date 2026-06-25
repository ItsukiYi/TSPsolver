"""A/B comparison: Original DualOpt vs Improved DualOpt.

Runs both versions on the same test instances and reports side-by-side results.

Usage:
    python compare_improvements.py --num-test 20
"""

import sys, os, argparse, time, json, math
import numpy as np
import torch

_project = os.path.dirname(__file__)

# LKH path
_lkh_dir = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7')
os.environ['PATH'] = _lkh_dir + os.pathsep + os.environ['PATH']


def load_data(num_test=20):
    """Load TSP-50 test instances."""
    instances = []
    with open(os.path.join(_project, 'data/tsp_problems/tsp50_test.txt')) as f:
        for i, line in enumerate(f):
            if i >= num_test: break
            line = line.strip()
            if not line: continue
            parts = line.split(' output ')
            coords = [float(x) for x in parts[0].split()]
            pts = np.array([[coords[j], coords[j+1]] for j in range(0, len(coords), 2)])
            instances.append(pts)
    return instances


def get_difusco_heatmap(points):
    """Run DIFUSCO inference to get edge probability heatmap.

    Args:
        points: (n, 2) numpy array

    Returns:
        heatmap: (n, n) numpy array of edge probabilities [0, 1]
    """
    # DIFUSCO paths must come BEFORE DualOpt and src to avoid 'utils' conflicts
    _saved_path = list(sys.path)
    _difusco = os.path.join(_project, 'DIFUSCO-main', 'difusco')
    _difusco_root = os.path.join(_project, 'DIFUSCO-main')
    # Remove conflicting paths temporarily
    _dualopt_paths = [p for p in sys.path if 'DualOpt' in p]
    for p in _dualopt_paths:
        sys.path.remove(p)
    _src_path = os.path.join(_project, 'src')
    if _src_path in sys.path:
        sys.path.remove(_src_path)
    # Insert DIFUSCO first
    sys.path.insert(0, _difusco)
    sys.path.insert(1, _difusco_root)
    # Clear cached 'utils' module (DualOpt's version was imported earlier)
    for key in list(sys.modules.keys()):
        if key.startswith('utils'):
            del sys.modules[key]

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
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name='heatmap')

    device = torch.device('cuda')
    ckpt = os.path.join(_project, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
    model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
    model = model.to(device); model.eval()

    n = len(points)
    with torch.no_grad():
        pts_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        heatmap = xt.float().cpu().numpy().squeeze() + 1e-6

    # Restore: put back DualOpt and src paths for downstream DualOpt evaluation
    sys.path[:] = _saved_path
    # Clear DIFUSCO's cached utils so DualOpt's utils loads fresh
    for key in list(sys.modules.keys()):
        if key.startswith('utils') or key == 'pl_tsp_model' or key == 'pl_meta_model':
            del sys.modules[key]
    # Ensure DualOpt and src are at front for subsequent DualOpt calls
    for p in _dualopt_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    if _src_path not in sys.path:
        sys.path.append(_src_path)
    return heatmap


def evaluate_dualopt(instances, dualopt_path, label, use_heatmap=False):
    """Run DualOpt (original or improved) on a list of instances.

    Args:
        use_heatmap: if True, passes DIFUSCO heatmap to second_step
    """
    # DualOpt path must come FIRST so its 'utils' takes priority
    old_path = list(sys.path)
    sys.path.insert(0, dualopt_path)
    # src path AFTER DualOpt (appended, not prepended)
    _src = os.path.join(_project, 'src')
    if _src not in sys.path:
        sys.path.append(_src)

    from utils import load_model
    from utils.functions import second_step, load_problem
    from src.algorithms import christofides_with_2opt

    # Load reviser models
    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(dualopt_path, f'pretrained/local_{size}/epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to('cuda'); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    class EvalOpts:
        revision_lens = [50, 20, 10]
        revision_iters = [25, 10, 5]
        problem = 'tsp'
        lkh_layer_number = 2

    opts = EvalOpts()
    get_cost = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

    _second_step = second_step  # save reference

    costs = []; times = []; heatmap_times = []
    print(f'\n[{label}]')
    for idx, pts in enumerate(instances):
        print(f'\r  {idx+1}/{len(instances)}', end='', flush=True)
        t0 = time.time()

        # Generate heatmap if requested
        heatmap = None
        if use_heatmap:
            ht0 = time.time()
            heatmap = get_difusco_heatmap(pts)
            heatmap_times.append(time.time() - ht0)
            # Re-import second_step from improved DualOpt (module cache was cleared)
            from utils.functions import second_step as _second_step_improved
            _second_step = _second_step_improved

        try:
            init_tour, _ = christofides_with_2opt(pts, max_2opt_iterations=100)
            if init_tour[-1] == init_tour[0]:
                init_tour = init_tour[:-1]
            seeds = torch.from_numpy(pts).float().unsqueeze(0).to('cuda')
            if use_heatmap:
                _, costs_revised = _second_step(seeds, get_cost, opts, revisers,
                                                heatmap=heatmap, tour_perm=init_tour)
            else:
                _, costs_revised = _second_step(seeds, get_cost, opts, revisers)
            cost = costs_revised.min().item()
        except Exception as e:
            print(f'\n  Error on instance {idx}: {e}')
            import traceback; traceback.print_exc()
            cost = float('nan')
        costs.append(cost)
        times.append(time.time() - t0)
    print()

    valid = [c for c in costs if not math.isnan(c)]
    result = {'mean': float(np.mean(valid)), 'std': float(np.std(valid)),
              'costs': costs, 'times': times, 'label': label,
              'success': len(valid), 'total': len(costs)}
    if heatmap_times:
        result['heatmap_time'] = float(np.mean(heatmap_times))
    sys.path[:] = old_path  # restore
    return result


def main():
    parser = argparse.ArgumentParser(description='A/B test: Original vs Improved DualOpt')
    parser.add_argument('--num-test', type=int, default=20)
    parser.add_argument('--output', type=str, default='outputs/improvement_comparison.json')
    args = parser.parse_args()

    instances = load_data(args.num_test)
    print(f'Loaded {len(instances)} TSP-50 test instances')
    print(f'{"=" * 60}')

    # Baseline: Original DualOpt (no heatmap)
    orig = evaluate_dualopt(
        instances,
        os.path.join(_project, 'DualOpt-main'),
        'Original DualOpt',
        use_heatmap=False
    )

    # Improved: DualOpt + DIFUSCO Heatmap Guidance
    impr = evaluate_dualopt(
        instances,
        os.path.join(_project, 'DualOpt-improved'),
        'DualOpt + Heatmap Guide',
        use_heatmap=True
    )

    # Summary
    print(f'\n{"=" * 60}')
    print(f'A/B COMPARISON')
    print(f'{"=" * 60}')
    print(f'{"Method":<25s} {"Mean":>8s} {"Std":>8s} {"Success":>8s}')
    print(f'{"-" * 50}')
    for r in [orig, impr]:
        print(f'{r["label"]:<25s} {r["mean"]:8.4f} {r["std"]:8.4f} {r["success"]:>4d}/{r["total"]}')

    if not math.isnan(orig['mean']) and not math.isnan(impr['mean']):
        delta = (impr['mean'] - orig['mean']) / orig['mean'] * 100
        better = 'improvement' if delta < 0 else 'degradation'
        print(f'\n  Change: {delta:+.2f}% ({better})')

        # Per-instance comparison
        print(f'\n  Per-instance deltas:')
        for i in range(len(instances)):
            if not math.isnan(orig['costs'][i]) and not math.isnan(impr['costs'][i]):
                d = (impr['costs'][i] - orig['costs'][i]) / orig['costs'][i] * 100
                marker = '++' if d < -2 else ('+' if d < 0 else ('-' if d < 2 else '--'))
                print(f'    Instance {i:2d}: orig={orig["costs"][i]:.4f} -> impr={impr["costs"][i]:.4f} ({d:+.2f}%) {marker}')

    # Save
    os.makedirs('outputs', exist_ok=True)
    output = {
        'original': {'mean': orig['mean'], 'std': orig['std'], 'costs': orig['costs'],
                     'times': orig['times'], 'success': orig['success']},
        'improved': {'mean': impr['mean'], 'std': impr['std'], 'costs': impr['costs'],
                     'times': impr['times'], 'success': impr['success']},
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
