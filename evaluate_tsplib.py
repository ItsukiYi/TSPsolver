"""Unified TSPLIB benchmark: compare all methods on standard instances.

Runs classic algorithms on TSPLIB instances and reports:
  - Solution cost
  - Gap to known optimal (%)
  - Runtime

Usage:
    python evaluate_tsplib.py                        # All classic methods
    python evaluate_tsplib.py --difusco <ckpt>       # Include DIFUSCO
    python evaluate_tsplib.py --dualopt               # Include DualOpt
    python evaluate_tsplib.py --all --ckpt <path>     # Everything
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.tsplib_loader import (
    load_tsplib_instance,
    list_tsplib_instances,
    KNOWN_OPTIMAL_COSTS,
)
from src.algorithms import (
    nearest_neighbor_tsp,
    christofides_tsp,
    christofides_with_2opt,
)
from src.utils import compute_distance_matrix, tour_cost


def evaluate_classic(points, opt_cost):
    """Run all classic algorithms and return results dict."""
    dist_mat = compute_distance_matrix(points)
    n = len(points)
    results = {}

    # 1. Nearest Neighbor
    t0 = time.perf_counter()
    tour, _ = nearest_neighbor_tsp(points)
    t1 = time.perf_counter()
    cost = tour_cost(dist_mat, tour)
    results["Nearest Neighbor"] = {
        "cost": float(cost),
        "time": t1 - t0,
        "gap_pct": float((cost / opt_cost - 1) * 100) if opt_cost else None,
    }

    # 2. Christofides
    t0 = time.perf_counter()
    tour, meta = christofides_tsp(points)
    t1 = time.perf_counter()
    cost = tour_cost(dist_mat, tour)
    results["Christofides"] = {
        "cost": float(cost),
        "time": t1 - t0,
        "gap_pct": float((cost / opt_cost - 1) * 100) if opt_cost else None,
    }

    # 3. Christofides + 2-opt (full)
    t0 = time.perf_counter()
    tour, meta = christofides_with_2opt(points, max_2opt_iterations=5000)
    t1 = time.perf_counter()
    cost = tour_cost(dist_mat, tour)
    results["Christofides+2opt"] = {
        "cost": float(cost),
        "time": t1 - t0,
        "gap_pct": float((cost / opt_cost - 1) * 100) if opt_cost else None,
    }

    return results


def evaluate_difusco(points, checkpoint_path, opt_cost=None):
    """Run DIFUSCO inference and return results."""
    import torch
    import numpy as np

    difusco_path = os.path.join(os.path.dirname(__file__), "DIFUSCO-main")
    sys.path.insert(0, difusco_path)
    sys.path.insert(0, os.path.join(difusco_path, "difusco"))

    from pl_tsp_model import TSPModel
    from argparse import Namespace
    from utils.tsp_utils import batched_two_opt_torch, merge_tours
    from utils.diffusion_schedulers import InferenceSchedule

    args = Namespace(
        diffusion_type="categorical", diffusion_schedule="cosine",
        diffusion_steps=1000, inference_diffusion_steps=50,
        inference_schedule="cosine", inference_trick="ddim",
        n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation="sum",
        two_opt_iterations=1000, parallel_sampling=1, sequential_sampling=1,
        save_numpy_heatmap=False,
        storage_path=os.getcwd(),
        training_split="dummy", validation_split="dummy", test_split="dummy",
        batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
        lr_scheduler="cosine-decay", num_epochs=50, num_workers=0,
        validation_examples=8, use_activation_checkpoint=False, fp16=False,
        project_name="eval", wandb_logger_name=None, wandb_entity=None,
        resume_id=None, ckpt_path=None,
        do_train=False, do_test=True, do_valid_only=False, resume_weight_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSPModel.load_from_checkpoint(checkpoint_path, param_args=args, strict=False)
    model = model.to(device)
    model.eval()

    n = len(points)
    n_batch = n

    t0 = time.perf_counter()

    with torch.no_grad():
        adj_matrix = np.zeros((1, n_batch, n_batch))
        points_t = torch.from_numpy(points).float().unsqueeze(0).to(device)

        xt = torch.randn_like(torch.from_numpy(adj_matrix).float()).to(device)
        xt = (xt > 0).long()

        steps = args.inference_diffusion_steps
        time_schedule = InferenceSchedule(
            inference_schedule=args.inference_schedule,
            T=model.diffusion.T, inference_T=steps,
        )

        for i in range(steps):
            t1_i, t2_i = time_schedule(i)
            t1_i = np.array([t1_i]).astype(int)
            t2_i = np.array([t2_i]).astype(int)
            xt = model.categorical_denoise_step(
                points_t, xt, t1_i, device, None, target_t=t2_i,
            )

        adj_mat = xt.float().cpu().numpy() + 1e-6

    np_points = points.astype(np.float64)
    tours, merge_iters = merge_tours(adj_mat, np_points, None, sparse_graph=False, parallel_sampling=1)
    solved_tours, ns = batched_two_opt_torch(
        np_points, np.array(tours).astype("int64"),
        max_iterations=args.two_opt_iterations, device=device,
    )

    t1 = time.perf_counter()
    tour = solved_tours[0].tolist()
    cost = tour_cost(compute_distance_matrix(points), tour)

    return {
        "cost": float(cost),
        "time": t1 - t0,
        "gap_pct": float((cost / opt_cost - 1) * 100) if opt_cost else None,
        "merge_iters": merge_iters,
        "2opt_iters": ns,
    }


def evaluate_dualopt(points, opt_cost=None):
    """Run DualOpt reviser and return results."""
    import torch

    dualopt_path = os.path.join(os.path.dirname(__file__), "DualOpt-main")
    if dualopt_path not in sys.path:
        sys.path.insert(0, dualopt_path)

    # Add LKH to PATH
    lkh_dir = os.path.join(dualopt_path, "LKH-3.0.7")
    os.environ["PATH"] = lkh_dir + os.pathsep + os.environ["PATH"]

    from utils import load_model
    from utils.functions import second_step, load_problem

    # Get initial solution
    _, meta = christofides_with_2opt(points, max_2opt_iterations=100)
    initial_tour = _last_tour  # saved from previous call

    # Actually we need to redo this properly
    tour_init, _ = christofides_with_2opt(points, max_2opt_iterations=100)
    if tour_init[-1] == tour_init[0]:
        tour_init = tour_init[:-1]

    seeds = torch.from_numpy(points).float().unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds.to(device)

    # Load revisers
    revisers = []
    for size in [50, 20, 10]:
        path = os.path.join(dualopt_path, f"pretrained/local_{size}/epoch-100.pt")
        reviser, _ = load_model(path, is_local=True)
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type("greedy")
        revisers.append(reviser)

    class EvalOpts:
        revision_lens = [50, 20, 10]
        revision_iters = [25, 10, 5]
        problem = "tsp"
        lkh_layer_number = 2

    opts = EvalOpts()
    get_cost = lambda inp, pi: load_problem("tsp").get_costs(inp, pi, return_local=True)

    t0 = time.perf_counter()
    _, costs_revised = second_step(seeds, get_cost, opts, revisers=revisers)
    t1 = time.perf_counter()

    cost = costs_revised.min().item()
    return {
        "cost": float(cost),
        "time": t1 - t0,
        "gap_pct": float((cost / opt_cost - 1) * 100) if opt_cost else None,
    }


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="TSPLIB Unified Benchmark")
    parser.add_argument("--difusco", type=str, default=None, help="DIFUSCO checkpoint path")
    parser.add_argument("--dualopt", action="store_true", help="Include DualOpt")
    parser.add_argument("--all", action="store_true", help="Run all methods")
    parser.add_argument("--instances", type=str, default=None,
                        help="Comma-separated instance names (default: all)")
    parser.add_argument("--output", type=str, default="outputs/tsplib_results.json")
    parser.add_argument("--data-dir", type=str, default="data/tsplib")

    args = parser.parse_args()

    if args.instances:
        instance_names = args.instances.split(",")
    else:
        instance_names = list_tsplib_instances(args.data_dir)

    print(f"{'=' * 70}")
    print(f"TSPLIB Unified Benchmark")
    print(f"{'=' * 70}")
    print(f"Instances: {instance_names}")
    methods = ["Nearest Neighbor", "Christofides", "Christofides+2opt"]
    if args.difusco or args.all:
        methods.append("DIFUSCO+2opt")
        print(f"DIFUSCO checkpoint: {args.difusco}")
    if args.dualopt or args.all:
        methods.append("DualOpt")
    print(f"Methods: {methods}")
    print(f"{'=' * 70}")

    all_results = {}

    for name in instance_names:
        print(f"\n--- {name} ---")
        points, _, opt_cost, display = load_tsplib_instance(name, args.data_dir)
        if opt_cost is None:
            print(f"  WARNING: no known optimum for {name}, skipping gap calculation")
            opt_cost = 0

        instance_results = {}

        # Classic methods (always run)
        classic = evaluate_classic(points, opt_cost)
        instance_results.update(classic)

        # DIFUSCO
        if args.difusco or args.all:
            try:
                difusco_res = evaluate_difusco(points, args.difusco or args.difusco, opt_cost)
                instance_results["DIFUSCO+2opt"] = difusco_res
            except Exception as e:
                print(f"  DIFUSCO failed: {e}")

        # DualOpt
        if args.dualopt or args.all:
            try:
                dualopt_res = evaluate_dualopt(points, opt_cost)
                instance_results["DualOpt"] = dualopt_res
            except Exception as e:
                print(f"  DualOpt failed: {e}")

        # Print results for this instance
        for method_name, res in instance_results.items():
            gap_str = f"gap={res['gap_pct']:.2f}%" if res['gap_pct'] is not None else ""
            print(f"  {method_name:25s}: cost={res['cost']:12.2f}  {gap_str}  ({res['time']:.2f}s)")

        all_results[name] = {"opt_cost": opt_cost, "methods": instance_results}

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Mean Gap to Optimal (%)")
    print(f"{'=' * 70}")
    header = f"{'Instance':<15s}"
    for m in methods:
        header += f"  {m:>16s}"
    print(header)
    print("-" * 70)

    for name in instance_names:
        row = f"{name:<15s}"
        for m in methods:
            res = all_results[name]["methods"].get(m, {})
            gap = res.get("gap_pct")
            if gap is not None:
                row += f"  {gap:15.2f}%"
            else:
                row += "  " + " " * 15
        print(row)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    serializable = {}
    for name, data in all_results.items():
        serializable[name] = {
            "opt_cost": data["opt_cost"],
            "methods": {
                m: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in res.items()}
                for m, res in data["methods"].items()
            }
        }
    with open(args.output, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
