"""Experiment runner for comparing TSP algorithms.

Runs classic algorithms (Nearest Neighbor, Christofides, Christofides+2opt)
on TSP instances of varying sizes, measures cost and runtime, and produces
comparison plots.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    compute_distance_matrix,
    generate_random_tsp_instance,
    generate_tsplib_style_instance,
    plot_tour,
    plot_comparison,
    Timer,
    run_experiment,
)
from .algorithms import (
    nearest_neighbor_tsp,
    christofides_tsp,
    two_opt_fast,
    two_opt_tsp,
    christofides_with_2opt,
)


# ============================================================
# Experiment Configuration
# ============================================================

DEFAULT_SIZES = [5, 10, 20, 50, 100, 200, 500]
DEFAULT_TRIALS = 5  # Number of instances per size
DEFAULT_OUTPUT_DIR = "../outputs"


def build_solvers(include_2opt: bool = True) -> Dict[str, Callable]:
    """Build the dictionary of solvers to compare.

    Args:
        include_2opt: whether to include standalone 2-opt (slow for n > 200)

    Returns:
        dict mapping solver name to solver function
    """
    solvers = {
        "Nearest Neighbor": lambda pts: nearest_neighbor_tsp(pts)[0],
        "Christofides": lambda pts: christofides_tsp(pts)[0],
        "Christofides + 2-opt": lambda pts: christofides_with_2opt(
            pts, max_2opt_iterations=1000, use_fast_2opt=True
        )[0],
    }

    if include_2opt:
        # Standalone 2-opt starting from random
        solvers["2-opt (from NN)"] = lambda pts: two_opt_fast(
            pts, max_iterations=1000
        )[0]

    return solvers


# ============================================================
# Main Experiment
# ============================================================

def run_full_experiment(
    sizes: List[int] = None,
    trials: int = DEFAULT_TRIALS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run a full experiment comparing all solvers on multiple instance sizes.

    Args:
        sizes: list of instance sizes to test
        trials: number of random instances per size
        output_dir: directory to save results
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        dict with all experimental results
    """
    if sizes is None:
        sizes = DEFAULT_SIZES

    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Generate instances
    instances_by_size = {}
    for n in sizes:
        instances_by_size[n] = [
            generate_random_tsp_instance(n, seed=seed + i) for i in range(trials)
        ]

    # Build solvers
    solvers = build_solvers(include_2opt=(max(sizes) <= 200))

    # Run experiments
    all_results = {}  # solver_name -> list of per-size results
    for solver_name, solver_fn in solvers.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running {solver_name}")
            print(f"{'=' * 60}")

        solver_results = []
        for n in sizes:
            if verbose:
                print(f"\n  Size n={n}:")

            results = run_experiment(
                solver_fn,
                instances_by_size[n],
                solver_name=f"{solver_name} (n={n})",
                verbose=verbose,
            )
            solver_results.append(results)

        all_results[solver_name] = solver_results

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"experiment_results_{timestamp}.json")

    # Convert numpy values for JSON serialization
    serializable_results = {}
    for solver_name, solver_results in all_results.items():
        serializable_results[solver_name] = []
        for r in solver_results:
            serializable_results[solver_name].append({
                "sizes": r["sizes"][0],  # all same size
                "mean_cost": float(np.mean(r["costs"])),
                "std_cost": float(np.std(r["costs"])),
                "mean_time": float(np.mean(r["times"])),
                "std_time": float(np.std(r["times"])),
                "raw_costs": [float(c) for c in r["costs"]],
                "raw_times": [float(t) for t in r["times"]],
            })

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {results_path}")

    return all_results


# ============================================================
# Visualization
# ============================================================

def plot_runtime_comparison(
    all_results: Dict,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save: bool = True,
):
    """Plot runtime vs instance size for all solvers.

    Args:
        all_results: results from run_full_experiment
        output_dir: directory for saving
        save: whether to save to file
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for (solver_name, solver_results), color in zip(all_results.items(), colors):
        sizes = [r["sizes"][0] for r in solver_results]
        mean_times = [np.mean(r["times"]) for r in solver_results]
        std_times = [np.std(r["times"]) for r in solver_results]

        ax.errorbar(
            sizes, mean_times, yerr=std_times,
            label=solver_name, color=color, marker="o",
            capsize=5, linewidth=2,
        )

    ax.set_xlabel("Number of Nodes (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime vs Instance Size", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(os.path.join(output_dir, f"runtime_comparison_{timestamp}.png"),
                    dpi=150, bbox_inches="tight")
    return fig


def plot_cost_comparison(
    all_results: Dict,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save: bool = True,
    include_optimal: bool = True,
):
    """Plot solution cost vs instance size.

    Args:
        all_results: results from run_full_experiment
        output_dir: directory for saving
        save: whether to save to file
        include_optimal: add reference line for optimal (sqrt(n)) scaling
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for (solver_name, solver_results), color in zip(all_results.items(), colors):
        sizes = [r["sizes"][0] for r in solver_results]
        mean_costs = [np.mean(r["costs"]) for r in solver_results]
        std_costs = [np.std(r["costs"]) for r in solver_results]

        ax.errorbar(
            sizes, mean_costs, yerr=std_costs,
            label=solver_name, color=color, marker="s",
            capsize=5, linewidth=2,
        )

    if include_optimal:
        # Expected optimal TSP tour length ~ 0.712 * sqrt(n) for unit square
        x = np.array(sizes)
        ax.plot(x, 0.712 * np.sqrt(x), "k--", alpha=0.4, label="~OPT (0.712√n)")

    ax.set_xlabel("Number of Nodes (n)", fontsize=12)
    ax.set_ylabel("Tour Cost", fontsize=12)
    ax.set_title("Solution Quality vs Instance Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(os.path.join(output_dir, f"cost_comparison_{timestamp}.png"),
                    dpi=150, bbox_inches="tight")
    return fig


def plot_approximation_ratio(
    all_results: Dict,
    baseline_solver: str = "Nearest Neighbor",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save: bool = True,
):
    """Plot approximation ratios relative to a baseline solver.

    Args:
        all_results: results from run_full_experiment
        baseline_solver: name of the baseline solver for ratio
        output_dir: directory for saving
        save: whether to save to file
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_results = all_results.get(baseline_solver)
    if baseline_results is None:
        print(f"Baseline '{baseline_solver}' not found in results")
        return fig

    baseline_costs = [np.mean(r["costs"]) for r in baseline_results]
    sizes = [r["sizes"][0] for r in baseline_results]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for (solver_name, solver_results), color in zip(all_results.items(), colors):
        if solver_name == baseline_solver:
            continue

        ratios = [
            np.mean(solver_results[i]["costs"]) / baseline_costs[i]
            for i in range(len(sizes))
        ]
        ax.plot(sizes, ratios, color=color, marker="o", linewidth=2, label=solver_name)

    # Christofides guarantee line
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.5, color="red", linestyle=":", alpha=0.5, label="Christofides guarantee (1.5)")

    ax.set_xlabel("Number of Nodes (n)", fontsize=12)
    ax.set_ylabel(f"Approximation Ratio vs {baseline_solver}", fontsize=12)
    ax.set_title(f"Algorithm Quality (relative to {baseline_solver})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(
            os.path.join(output_dir, f"approx_ratio_{timestamp}.png"),
            dpi=150, bbox_inches="tight",
        )
    return fig
