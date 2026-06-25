"""Final comparison: DIFUSCO + DualOpt + Classic on TSP-50 test set.

Uses the same test file for all methods. Fair comparison.

Usage:
    python final_comparison.py --ckpt tsp50_categorical/checkpoints/epoch=6-step=105.ckpt
    python final_comparison.py --ckpt ... --dualopt  # include DualOpt
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import numpy as np

# Set path order carefully
_project = os.path.dirname(__file__)
_difusco_path = os.path.join(_project, "DIFUSCO-main")
_src_path = os.path.join(_project, "src")
_dualopt_path = os.path.join(_project, "DualOpt-main")

# Only add DIFUSCO and src at module level (DualOpt conflicts with both)
if _difusco_path not in sys.path:
    sys.path.insert(0, os.path.join(_difusco_path, "difusco"))
    sys.path.insert(1, _difusco_path)
if _src_path not in sys.path:
    sys.path.insert(2, _src_path)

from src.utils import compute_distance_matrix, tour_cost
from src.algorithms import nearest_neighbor_tsp, christofides_tsp, christofides_with_2opt


# ============================================================
# Data Loading
# ============================================================

def load_test_data(filepath, max_instances=None):
    """Load TSP instances from DIFUSCO-format file."""
    instances = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" output ")
            coords = [float(x) for x in parts[0].split()]
            points = np.array([[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)])
            tour = [int(t) - 1 for t in parts[1].split()]
            instances.append((points, tour))
            if max_instances and len(instances) >= max_instances:
                break
    return instances


# ============================================================
# DIFUSCO Inference
# ============================================================

def run_difusco(checkpoint_path, test_instances, two_opt_iterations=1000):
    """Run DIFUSCO inference on test instances."""
    from pl_tsp_model import TSPModel
    from argparse import Namespace
    import torch
    from utils.tsp_utils import batched_two_opt_torch, merge_tours
    from utils.diffusion_schedulers import InferenceSchedule

    args = Namespace(
        diffusion_type="categorical", diffusion_schedule="cosine",
        diffusion_steps=1000, inference_diffusion_steps=50,
        inference_schedule="cosine", inference_trick="ddim",
        n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation="sum",
        two_opt_iterations=two_opt_iterations,
        parallel_sampling=1, sequential_sampling=1,
        save_numpy_heatmap=False,
        storage_path=".",
        training_split="data/tsp_problems/tsp50_test.txt",
        validation_split="data/tsp_problems/tsp50_test.txt",
        test_split="data/tsp_problems/tsp50_test.txt",
        batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
        lr_scheduler="cosine-decay", num_epochs=50, num_workers=0,
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name="eval", wandb_logger_name=None, wandb_entity=None,
        resume_id=None, ckpt_path=None,
        do_train=False, do_test=True, do_valid_only=False, resume_weight_only=True,
    )

    device = torch.device("cuda")
    model = TSPModel.load_from_checkpoint(checkpoint_path, param_args=args, strict=False)
    model = model.to(device)
    model.eval()

    results = []
    for idx, (points, _) in enumerate(test_instances):
        print(f"\r  DIFUSCO: {idx + 1}/{len(test_instances)}", end="", flush=True)

        n = len(points)
        with torch.no_grad():
            points_t = torch.from_numpy(points).float().unsqueeze(0).to(device)
            adj_init = np.zeros((1, n, n))
            xt = torch.randn_like(torch.from_numpy(adj_init).float()).to(device)
            xt = (xt > 0).long()

            steps = args.inference_diffusion_steps
            time_schedule = InferenceSchedule(
                inference_schedule=args.inference_schedule,
                T=model.diffusion.T, inference_T=steps,
            )
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                xt = model.categorical_denoise_step(points_t, xt, t1, device, None, target_t=t2)

            adj_mat = xt.float().cpu().numpy() + 1e-6

        np_points = points.astype(np.float64)
        tours, merge_iters = merge_tours(adj_mat, np_points, None, sparse_graph=False, parallel_sampling=1)
        solved_tours, ns = batched_two_opt_torch(
            np_points, np.array(tours).astype("int64"),
            max_iterations=args.two_opt_iterations, device=device,
        )
        tour = solved_tours[0].tolist()
        cost = tour_cost(compute_distance_matrix(points), tour)
        results.append({"cost": cost, "merge_iters": merge_iters, "2opt_iters": ns})

    print()
    return results


# ============================================================
# DualOpt Inference
# ============================================================

def run_dualopt(test_instances):
    """Run DualOpt reviser on test instances."""
    import torch

    # Temporarily swap DualOpt path to front for its imports
    _old_path = list(sys.path)
    if _dualopt_path in sys.path:
        sys.path.remove(_dualopt_path)
    sys.path.insert(0, _dualopt_path)

    from utils import load_model
    from utils.functions import second_step, load_problem

    # Restore original path order
    sys.path[:] = _old_path

    # LKH path
    lkh_dir = os.path.join(_dualopt_path, "LKH-3.0.7")
    os.environ["PATH"] = lkh_dir + os.pathsep + os.environ["PATH"]

    # Load reviser models (load once)
    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(_dualopt_path, f"pretrained/local_{size}/epoch-100.pt")
        r, _ = load_model(path, is_local=True)
        r.to("cuda")
        r.eval()
        r.set_decode_type("greedy")
        revisers.append(r)

    class EvalOpts:
        revision_lens = [50, 20, 10]
        revision_iters = [25, 10, 5]
        problem = "tsp"
        lkh_layer_number = 2

    opts = EvalOpts()
    get_cost = lambda inp, pi: load_problem("tsp").get_costs(inp, pi, return_local=True)

    results = []
    for idx, (points, _) in enumerate(test_instances):
        print(f"\r  DualOpt: {idx + 1}/{len(test_instances)}", end="", flush=True)

        # Initial: light Christofides+2opt
        init_tour, _ = christofides_with_2opt(points, max_2opt_iterations=100)
        if init_tour[-1] == init_tour[0]:
            init_tour = init_tour[:-1]

        seeds = torch.from_numpy(points).float().unsqueeze(0).to("cuda")
        t0 = time.perf_counter()
        _, costs_revised = second_step(seeds, get_cost, opts, revisers=revisers)
        elapsed = time.perf_counter() - t0

        cost = costs_revised.min().item()
        results.append({"cost": cost, "time": elapsed})

    print()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TSP-50 Unified Benchmark")
    parser.add_argument("--ckpt", type=str, required=True, help="DIFUSCO checkpoint")
    parser.add_argument("--dualopt", action="store_true", help="Include DualOpt")
    parser.add_argument("--test-file", type=str, default="data/tsp_problems/tsp50_test.txt")
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    test_data = load_test_data(args.test_file, max_instances=args.num_test)
    test_points = [p for p, _ in test_data]
    test_tours = [t for _, t in test_data]
    n_instances = len(test_data)
    n_nodes = len(test_data[0][0])

    print(f"\n{'=' * 65}")
    print(f"  TSP-{n_nodes} Unified Benchmark ({n_instances} instances)")
    print(f"{'=' * 65}")
    print(f"  Test file: {args.test_file}")
    print(f"  Methods:   NN | Christofides | C+2opt | DIFUSCO+2opt", end="")
    if args.dualopt:
        print(" | DualOpt", end="")
    print()
    print(f"{'=' * 65}")

    all_results = {}

    # ---- Classic Baselines ----
    for label, solver_fn in [
        ("Nearest Neighbor", lambda pts: nearest_neighbor_tsp(pts)[0]),
        ("Christofides", lambda pts: christofides_tsp(pts)[0]),
    ]:
        print(f"\n[{label}]")
        costs = []
        for i, pts in enumerate(test_points):
            print(f"\r  {i + 1}/{n_instances}", end="", flush=True)
            tour = solver_fn(pts)
            costs.append(tour_cost(compute_distance_matrix(pts), tour))
        print()
        all_results[label] = {"costs": costs, "mean": np.mean(costs), "std": np.std(costs)}

    # ---- Christofides + 2-opt (baseline) ----
    print("\n[Christofides + 2-opt]")
    ch2_costs = []
    for i, pts in enumerate(test_points):
        print(f"\r  {i + 1}/{n_instances}", end="", flush=True)
        tour, _ = christofides_with_2opt(pts, max_2opt_iterations=1000)
        ch2_costs.append(tour_cost(compute_distance_matrix(pts), tour))
    print()
    all_results["Christofides + 2-opt"] = {"costs": ch2_costs, "mean": np.mean(ch2_costs), "std": np.std(ch2_costs)}

    # ---- DIFUSCO ----
    print("\n[DIFUSCO + 2-opt]")
    dif_res = run_difusco(args.ckpt, test_data)
    dif_costs = [r["cost"] for r in dif_res]
    all_results["DIFUSCO + 2-opt"] = {"costs": dif_costs, "mean": np.mean(dif_costs), "std": np.std(dif_costs)}

    # ---- DualOpt ----
    if args.dualopt:
        print("\n[DualOpt]")
        do_res = run_dualopt(test_data)
        do_costs = [r["cost"] for r in do_res]
        all_results["DualOpt"] = {"costs": do_costs, "mean": np.mean(do_costs), "std": np.std(do_costs)}

    # ---- Summary ----
    nn_mean = all_results["Nearest Neighbor"]["mean"]
    ch2_mean = all_results["Christofides + 2-opt"]["mean"]

    print(f"\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Method':<25s} {'Mean':>8s} {'Std':>8s} {'vs NN':>8s} {'vs C+2opt':>10s}")
    print(f"  {'-' * 55}")

    for name in all_results:
        m = all_results[name]["mean"]
        s = all_results[name]["std"]
        vs_nn = (m / nn_mean - 1) * 100
        vs_ch2 = (m / ch2_mean - 1) * 100
        print(f"  {name:<25s} {m:8.4f} {s:8.4f} {vs_nn:+7.1f}% {vs_ch2:+9.1f}%")

    # Also compute gap to ground truth (Christofides+2opt 5000-iter labels)
    gt_costs = [tour_cost(compute_distance_matrix(pts), tour) for pts, tour in test_data]
    gt_mean = np.mean(gt_costs)
    print(f"\n  Ground truth (C+2opt-5000): {gt_mean:.4f}")
    print(f"  {'Method':<25s} {'vs GT':>8s}")
    print(f"  {'-' * 35}")
    for name in all_results:
        gap = (all_results[name]["mean"] / gt_mean - 1) * 100
        print(f"  {name:<25s} {gap:+7.2f}%")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "config": {"n": n_nodes, "n_instances": n_instances, "test_file": args.test_file},
        "ground_truth_mean": float(gt_mean),
        "results": {
            name: {"mean": float(v["mean"]), "std": float(v["std"]),
                   "costs": [float(c) for c in v["costs"]]}
            for name, v in all_results.items()
        },
    }
    path = os.path.join(args.output_dir, f"tsp{n_nodes}_benchmark_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
