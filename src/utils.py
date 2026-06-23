"""TSP Utilities: data generation, distance computation, evaluation, visualization."""

import math
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


# ============================================================
# Distance / Cost Functions
# ============================================================

def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix for a set of 2D points.

    Args:
        points: numpy array of shape (n, 2)

    Returns:
        dist_mat: numpy array of shape (n, n) with Euclidean distances
    """
    return distance_matrix(points, points)


def tour_cost(dist_mat: np.ndarray, tour: List[int]) -> float:
    """Compute the total cost (distance) of a tour.

    Args:
        dist_mat: (n, n) distance matrix
        tour: list of vertex indices representing the tour

    Returns:
        total distance as float
    """
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += dist_mat[tour[i], tour[i + 1]]
    return cost


# ============================================================
# TSP Instance Generation
# ============================================================

def generate_random_tsp_instance(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random TSP instance with n points in the unit square [0, 1]^2.

    Args:
        n: number of points
        seed: random seed for reproducibility

    Returns:
        points: numpy array of shape (n, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n, 2)


def generate_tsplib_style_instance(
    n: int,
    distribution: str = "uniform",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a TSP instance in the style of TSPLIB benchmarks.

    Args:
        n: number of points
        distribution: "uniform" or "clustered" (clustered generates
                      more realistic delivery-scenario data)
        seed: random seed

    Returns:
        points: numpy array of shape (n, 2)
    """
    if seed is not None:
        np.random.seed(seed)

    if distribution == "uniform":
        return np.random.rand(n, 2)
    elif distribution == "clustered":
        # Generate clustered points to simulate delivery zones
        n_clusters = int(np.sqrt(n))
        points = []
        for _ in range(n_clusters):
            center = np.random.rand(2)
            cluster_size = n // n_clusters
            cluster_points = center + 0.05 * np.random.randn(cluster_size, 2)
            cluster_points = np.clip(cluster_points, 0, 1)
            points.append(cluster_points)
        # Fill remaining
        remaining = n - sum(p.shape[0] for p in points)
        if remaining > 0:
            points.append(np.random.rand(remaining, 2))
        return np.concatenate(points, axis=0)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ============================================================
# Optimal Solution (for small instances, via brute force)
# ============================================================

def brute_force_tsp(points: np.ndarray) -> Tuple[List[int], float]:
    """Compute optimal TSP tour via brute force. Only for n <= 10!

    Args:
        points: numpy array of shape (n, 2)

    Returns:
        (tour, cost) tuple
    """
    n = len(points)
    assert n <= 10, f"Brute force only feasible for n <= 10, got n = {n}"
    dist_mat = compute_distance_matrix(points)

    best_tour = None
    best_cost = float("inf")

    # Fix start at 0, permute remaining n-1 vertices
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm) + [0]
        cost = tour_cost(dist_mat, tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost


# ============================================================
# Visualization
# ============================================================

def plot_tour(
    points: np.ndarray,
    tour: List[int],
    title: str = "TSP Tour",
    figsize: Tuple[int, int] = (8, 6),
    show_indices: bool = False,
) -> plt.Figure:
    """Plot a TSP tour on 2D coordinates.

    Args:
        points: numpy array of shape (n, 2)
        tour: list of vertex indices
        title: plot title
        figsize: figure size
        show_indices: whether to show vertex indices

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c="blue", s=50, zorder=2)

    # Mark depot (start/end point)
    ax.scatter(*points[tour[0]], c="red", s=100, marker="*", zorder=3, label="Depot")

    # Plot tour edges
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        ax.plot(
            [points[u, 0], points[v, 0]],
            [points[u, 1], points[v, 1]],
            "b-",
            alpha=0.6,
            linewidth=1.5,
        )

    if show_indices:
        for i, (x, y) in enumerate(points):
            ax.annotate(str(i), (x, y), fontsize=8, ha="right")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_comparison(
    points: np.ndarray,
    tours: List[Tuple[List[int], str]],
    figsize: Tuple[int, int] = (16, 5),
):
    """Plot multiple tours side by side for comparison.

    Args:
        points: numpy array of shape (n, 2)
        tours: list of (tour, label) tuples
        figsize: figure size
    """
    n_tours = len(tours)
    fig, axes = plt.subplots(1, n_tours, figsize=figsize)

    for ax, (tour, label) in zip(axes, tours):
        ax.scatter(points[:, 0], points[:, 1], c="blue", s=30)
        ax.scatter(*points[tour[0]], c="red", s=80, marker="*")
        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i + 1]
            ax.plot(
                [points[u, 0], points[v, 0]],
                [points[u, 1], points[v, 1]],
                "b-",
                alpha=0.6,
                linewidth=1.0,
            )
        cost = tour_cost(compute_distance_matrix(points), tour)
        ax.set_title(f"{label}\nCost: {cost:.4f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# Experiment Utilities
# ============================================================

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def run_experiment(
    solver_fn,
    instances: List[np.ndarray],
    solver_name: str = "Solver",
    verbose: bool = True,
) -> dict:
    """Run a solver on a list of TSP instances and collect results.

    Args:
        solver_fn: function that takes points and returns (tour, metadata)
        instances: list of point arrays
        solver_name: name for logging
        verbose: whether to print progress

    Returns:
        dict with keys: costs, times, sizes, solver_name
    """
    costs = []
    times = []
    sizes = []

    for i, points in enumerate(instances):
        if verbose:
            print(f"\r{solver_name}: {i + 1}/{len(instances)}", end="", flush=True)

        timer = Timer()
        with timer:
            result = solver_fn(points)
        elapsed = timer.elapsed

        # Handle different return types
        if isinstance(result, tuple):
            tour = result[0]
        else:
            tour = result

        dist_mat = compute_distance_matrix(points)
        cost = tour_cost(dist_mat, tour)

        costs.append(cost)
        times.append(elapsed)
        sizes.append(len(points))

    if verbose:
        print()  # newline

    return {
        "solver_name": solver_name,
        "costs": costs,
        "times": times,
        "sizes": sizes,
    }
