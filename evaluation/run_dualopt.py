"""Run DualOpt evaluation on TSP-50 test data.

Usage:
    python run_dualopt.py --test-file data/tsp_problems/tsp50_test.txt --num-samples 10
"""

import sys
import os
import argparse

import numpy as np
import torch

# ---- Setup: DualOpt path (with its own 'utils' package) ----
_script_dir = os.path.dirname(__file__)
_dualopt_path = os.path.join(_script_dir, "DualOpt-main")
_src_path = os.path.join(_script_dir, "src")

# Add DualOpt first, src LAST to avoid utils name clash
if _dualopt_path not in sys.path:
    sys.path.insert(0, _dualopt_path)
if _src_path not in sys.path:
    sys.path.append(_src_path)  # append so DualOpt's utils takes priority

# Import DualOpt modules
from utils import load_model
from utils.functions import second_step, load_problem

# Import our src modules - must use src. prefix since algorithms.py uses relative imports
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

# Add LKH to PATH
_lkh_dir = os.path.join(_dualopt_path, "LKH-3.0.7")
os.environ["PATH"] = _lkh_dir + os.pathsep + os.environ["PATH"]

# We need to alias our src/utils.py to avoid conflict with DualOpt's utils
# This was handled by renaming the import above


def read_difusco_data(filepath):
    """Read TSP data in DIFUSCO format (same as DIFUSCO)."""
    instances = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" output ")
            coords = [float(x) for x in parts[0].split()]
            points = np.array([[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)])
            tour = [int(t) - 1 for t in parts[1].split()]  # 0-indexed
            instances.append((points, tour))
    return instances


def run_dualopt_reviser(points, initial_tour, device="cuda"):
    """Run DualOpt's neural reviser models to refine an initial tour."""
    reviser_sizes = [50, 20, 10]

    revisers = []
    for size in reviser_sizes:
        reviser_path = os.path.join(_dualopt_path, f"pretrained/local_{size}/epoch-100.pt")
        if not os.path.exists(reviser_path):
            print(f"  Warning: reviser size={size} not found")
            continue
        reviser, _ = load_model(reviser_path, is_local=True)
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type("greedy")
        revisers.append(reviser)

    if not revisers:
        raise RuntimeError("No reviser models loaded")

    if initial_tour[0] == initial_tour[-1]:
        initial_tour = initial_tour[:-1]

    seeds = torch.from_numpy(points).float().unsqueeze(0).to(device)

    class EvalOpts:
        revision_lens = [50, 20, 10]
        revision_iters = [25, 10, 5]
        problem = "tsp"  # Must be 'tsp' for second_step to work correctly
        lkh_layer_number = 2

    opts = EvalOpts()
    # Use TSP problem type which supports return_local=True
    get_cost_func = lambda inp, pi: load_problem("tsp").get_costs(inp, pi, return_local=True)
    costs_original, costs_revised = second_step(seeds, get_cost_func, opts, revisers=revisers)

    return costs_original.min().item(), costs_revised.min().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, default="data/tsp_problems/tsp50_test.txt")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("DualOpt Neural Reviser Evaluation (TSP-50)")
    print(f"{'=' * 60}")
    print(f"  Test: {args.test_file}  |  Samples: {args.num_samples}  |  GPU: {args.device}")
    print(f"{'=' * 60}")

    instances = read_difusco_data(args.test_file)[:args.num_samples]
    print(f"Loaded {len(instances)} instances")

    results = []
    for idx, (points, gt_tour) in enumerate(instances):
        print(f"\n[{idx + 1}/{len(instances)}] TSP-{len(points)}...")

        # Use light Christofides+2opt as initial (room for reviser to improve)
        initial_tour, _ = christofides_with_2opt(points, max_2opt_iterations=100)
        dist_mat = compute_distance_matrix(points)
        init_cost = tour_cost(dist_mat, initial_tour)
        gt_cost = tour_cost(dist_mat, gt_tour)

        try:
            _, refined_cost = run_dualopt_reviser(points, initial_tour, device=args.device)
            impr = (init_cost - refined_cost) / init_cost * 100
            print(f"  Init: {init_cost:.4f} -> Reviser: {refined_cost:.4f} ({impr:+.2f}%)  [GT: {gt_cost:.4f}]")
        except Exception as e:
            print(f"  Reviser error: {e}")
            import traceback; traceback.print_exc()
            refined_cost = init_cost

        results.append({"init": float(init_cost), "refined": float(refined_cost), "gt": float(gt_cost)})

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    if results:
        init_c = [r["init"] for r in results]
        ref_c = [r["refined"] for r in results]
        gt_c = [r["gt"] for r in results]
        print(f"  Initial (C+2opt-lite):  {np.mean(init_c):.4f} +- {np.std(init_c):.4f}")
        print(f"  After DualOpt reviser:  {np.mean(ref_c):.4f} +- {np.std(ref_c):.4f}")
        print(f"  Ground truth (full):    {np.mean(gt_c):.4f} +- {np.std(gt_c):.4f}")
        delta = (np.mean(ref_c) / np.mean(init_c) - 1) * 100
        print(f"  Reviser change:         {delta:+.2f}%")


if __name__ == "__main__":
    main()
