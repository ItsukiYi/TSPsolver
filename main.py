"""CS240 Project: Delivery Route Optimization in Modern Logistics
Main entry point for running experiments.

Usage:
    # Quick test of all algorithms on a single instance
    python main.py --mode demo --n 50

    # Run full benchmark experiment
    python main.py --mode experiment --sizes 10,20,50,100,200 --trials 5

    # Run DIFUSCO inference (requires pretrained checkpoint)
    python main.py --mode difusco --n 50 --ckpt <path_to_checkpoint>

    # Generate dataset for DIFUSCO training
    python main.py --mode generate-data --n 50 --num-samples 100
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.utils import (
    generate_random_tsp_instance,
    compute_distance_matrix,
    tour_cost,
    plot_tour,
    plot_comparison,
)
from src.algorithms import (
    nearest_neighbor_tsp,
    christofides_tsp,
    two_opt_fast,
    christofides_with_2opt,
)
from src.experiment import (
    run_full_experiment,
    plot_runtime_comparison,
    plot_cost_comparison,
    plot_approximation_ratio,
)

warnings.filterwarnings("ignore")


# ============================================================
# Demo mode: visualize algorithms on a single instance
# ============================================================

def demo(n: int = 30, seed: int = 42, output_dir: str = "outputs"):
    """Run a demo of all classic algorithms on a single instance."""
    print(f"\n{'=' * 60}")
    print(f"CS240 TSP Solver Demo (n={n})")
    print(f"{'=' * 60}")

    # Generate instance
    points = generate_random_tsp_instance(n, seed=seed)
    dist_mat = compute_distance_matrix(points)
    print(f"\nGenerated {n} random points in [0,1]^2")

    # Run each algorithm
    algorithms = {}

    print("\n[1/4] Running Nearest Neighbor...")
    nn_tour, nn_meta = nearest_neighbor_tsp(points)
    nn_cost = tour_cost(dist_mat, nn_tour)
    algorithms["Nearest Neighbor"] = (nn_tour, nn_cost)
    print(f"      Cost: {nn_cost:.4f}")

    print("[2/4] Running Christofides...")
    ch_tour, ch_meta = christofides_tsp(points)
    ch_cost = tour_cost(dist_mat, ch_tour)
    algorithms["Christofides"] = (ch_tour, ch_cost)
    print(f"      Cost: {ch_cost:.4f}  (MST weight: {ch_meta['mst_weight']:.4f}, "
          f"odd vertices: {ch_meta['num_odd_vertices']})")

    print("[3/4] Running Christofides + 2-opt...")
    ch2_tour, ch2_meta = christofides_with_2opt(points, max_2opt_iterations=1000)
    ch2_cost = tour_cost(dist_mat, ch2_tour)
    algorithms["Christofides + 2-opt"] = (ch2_tour, ch2_cost)
    print(f"      Cost: {ch2_cost:.4f}  "
          f"(improvement: {ch2_meta['improvement_pct']:.1f}%, "
          f"2-opt iters: {ch2_meta['two_opt_iterations']})")

    print("[4/4] Running 2-opt (from NN)...")
    opt2_tour, opt2_meta = two_opt_fast(points, max_iterations=1000)
    opt2_cost = tour_cost(dist_mat, opt2_tour)
    algorithms["2-opt (from NN)"] = (opt2_tour, opt2_cost)
    print(f"      Cost: {opt2_cost:.4f}  (iterations: {opt2_meta['iterations']})")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for name, (tour, cost) in algorithms.items():
        improvement = 100 * (cost - nn_cost) / nn_cost if name != "Nearest Neighbor" else 0
        improvement_str = f"({improvement:+.1f}% vs NN)" if name != "Nearest Neighbor" else "(baseline)"
        print(f"  {name:25s}: {cost:.4f}  {improvement_str}")

    # Visualize
    os.makedirs(output_dir, exist_ok=True)

    # Individual plots
    for name, (tour, cost) in algorithms.items():
        fig = plot_tour(points, tour, title=f"{name} (cost={cost:.4f})")
        fig.savefig(os.path.join(output_dir, f"demo_{name.replace(' ', '_')}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Comparison plot
    tours_for_comparison = [(tour, name) for name, (tour, _) in algorithms.items()]
    fig = plot_comparison(points, tours_for_comparison, figsize=(16, 8))
    fig.savefig(os.path.join(output_dir, "demo_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {output_dir}/")
    return algorithms


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CS240: Delivery Route Optimization — TSP Solver Benchmarks"
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["demo", "experiment", "difusco", "generate-data"],
        help="Run mode"
    )
    parser.add_argument("--n", type=int, default=30, help="Number of nodes for demo/difusco")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--sizes", type=str, default="10,20,50,100,200,500",
                        help="Comma-separated list of sizes for experiment mode")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per size for experiment mode")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to DIFUSCO checkpoint (for difusco mode)")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples for generate-data mode")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "demo":
        demo(n=args.n, seed=args.seed, output_dir=args.output_dir)

    elif args.mode == "experiment":
        sizes = [int(s) for s in args.sizes.split(",")]
        print(f"Running full experiment: sizes={sizes}, trials={args.trials}")
        results = run_full_experiment(
            sizes=sizes,
            trials=args.trials,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        if not args.no_plot:
            print("\nGenerating plots...")
            fig1 = plot_runtime_comparison(results, output_dir=args.output_dir)
            plt.close(fig1)
            fig2 = plot_cost_comparison(results, output_dir=args.output_dir)
            plt.close(fig2)
            fig3 = plot_approximation_ratio(results, output_dir=args.output_dir)
            plt.close(fig3)
            print(f"Plots saved to {args.output_dir}/")

    elif args.mode == "difusco":
        from src.difusco_wrapper import DIFUSCOInference
        if args.ckpt is None:
            print("ERROR: --ckpt is required for difusco mode")
            print("Download pretrained checkpoints from:")
            print("  https://drive.google.com/drive/folders/1IjaWtkqTAs7lwtFZ24lTRspE0h1N6sBH")
            sys.exit(1)

        points = generate_random_tsp_instance(args.n, seed=args.seed)
        print(f"Running DIFUSCO on TSP-{args.n} instance...")

        solver = DIFUSCOInference(checkpoint_path=args.ckpt)
        tour, meta = solver.solve(points)

        dist_mat = compute_distance_matrix(points)
        cost = tour_cost(dist_mat, tour)
        print(f"DIFUSCO cost: {cost:.4f}")

        # Also run baseline for comparison
        _, ch2_meta = christofides_with_2opt(points)
        print(f"Christofides+2opt cost: {ch2_meta['refined_cost']:.4f}")

        if not args.no_plot:
            fig = plot_tour(points, tour, title=f"DIFUSCO (cost={cost:.4f})")
            fig.savefig(os.path.join(args.output_dir, "difusco_result.png"), dpi=150)

    elif args.mode == "generate-data":
        from src.difusco_wrapper import generate_difusco_dataset
        from src.algorithms import christofides_with_2opt

        print(f"Generating {args.num_samples} TSP-{args.n} instances...")
        instances = []
        for i in range(args.num_samples):
            if i % 50 == 0:
                print(f"  Generating... {i}/{args.num_samples}")
            points = generate_random_tsp_instance(args.n, seed=args.seed + i)
            tour, _ = christofides_with_2opt(points, max_2opt_iterations=5000)
            instances.append((points, tour))

        output_file = os.path.join(args.output_dir, f"tsp{args.n}_christofides_2opt.txt")
        generate_difusco_dataset(instances, output_file)


if __name__ == "__main__":
    main()
