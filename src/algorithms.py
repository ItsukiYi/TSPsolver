"""Classic TSP Algorithms: Nearest Neighbor, Christofides, and 2-opt local search.

This module implements the baseline algorithms specified in the CS240 proposal:
1. Nearest Neighbor Greedy (O(n²))
2. Christofides Algorithm (1.5-approximation for Metric TSP)
3. 2-opt Local Search (post-processing refinement)
"""

import heapq
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx

from .utils import compute_distance_matrix, tour_cost


# ============================================================
# 1. Nearest Neighbor Greedy Algorithm  (O(n²))
# ============================================================

def nearest_neighbor_tsp(points: np.ndarray, start_node: int = 0) -> Tuple[List[int], dict]:
    """Solve TSP using the Nearest Neighbor greedy heuristic.

    Algorithm:
    1. Start at the start_node
    2. Repeatedly visit the nearest unvisited vertex
    3. Return to start_node at the end

    Complexity: O(n²) time, O(n) space

    Args:
        points: numpy array of shape (n, 2)
        start_node: index of the starting vertex (default 0 = depot)

    Returns:
        (tour, metadata) where tour is a list of vertex indices
    """
    n = len(points)
    dist_mat = compute_distance_matrix(points)

    visited = [False] * n
    tour = [start_node]
    visited[start_node] = True

    current = start_node
    for _ in range(n - 1):
        # Find the nearest unvisited vertex
        min_dist = float("inf")
        nearest = None
        for j in range(n):
            if not visited[j] and dist_mat[current, j] < min_dist:
                min_dist = dist_mat[current, j]
                nearest = j

        tour.append(nearest)
        visited[nearest] = True
        current = nearest

    # Return to start
    tour.append(start_node)

    metadata = {"algorithm": "NearestNeighbor"}
    return tour, metadata


# ============================================================
# 2. Christofides Algorithm  (1.5-approximation)
# ============================================================

def christofides_tsp(points: np.ndarray) -> Tuple[List[int], dict]:
    """Solve Metric TSP using Christofides' 1.5-approximation algorithm.

    Algorithm steps:
    1. Compute the Minimum Spanning Tree (MST) of the complete graph
    2. Find all vertices with odd degree in the MST
    3. Compute a Minimum-Weight Perfect Matching (MWPM) on the odd-degree vertices
    4. Combine MST + MWPM edges → Eulerian multigraph
    5. Find an Eulerian circuit in the combined graph
    6. Shortcut the circuit (skip visited vertices) to get a Hamiltonian cycle

    Theoretical guarantee: cost <= 1.5 * OPT for Metric TSP

    Complexity: O(n³) due to the MWPM step

    Args:
        points: numpy array of shape (n, 2)

    Returns:
        (tour, metadata) where tour is a list of vertex indices
    """
    n = len(points)
    dist_mat = compute_distance_matrix(points)

    # ---- Step 1: Minimum Spanning Tree ----
    # Use scipy's sparse MST (returns a sparse matrix)
    mst_sparse = minimum_spanning_tree(dist_mat)
    mst_graph = nx.from_scipy_sparse_array(mst_sparse)
    mst_weight = mst_sparse.sum()

    # ---- Step 2: Find odd-degree vertices ----
    odd_vertices = [v for v in range(n) if mst_graph.degree(v) % 2 == 1]

    # ---- Step 3: Minimum-Weight Perfect Matching on odd-degree vertices ----
    # Build a complete graph on the odd-degree vertices
    G_odd = nx.Graph()
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            if i < j:
                G_odd.add_edge(u, v, weight=-dist_mat[u, v])  # negative for max-weight matching
    # Use Blossom algorithm (nx.max_weight_matching) for MWPM
    # We negate weights above so max_weight_matching gives min weight matching
    matching = nx.max_weight_matching(G_odd, maxcardinality=True)

    # ---- Step 4: Combine MST + Matching → Eulerian graph ----
    eulerian_graph = nx.MultiGraph()
    # Add MST edges
    for u, v in mst_graph.edges():
        eulerian_graph.add_edge(u, v, weight=dist_mat[u, v])
    # Add matching edges
    for u, v in matching:
        eulerian_graph.add_edge(u, v, weight=dist_mat[u, v])

    # ---- Step 5: Find Eulerian Circuit ----
    # nx.eulerian_circuit returns a list of edges
    eulerian_edges = list(nx.eulerian_circuit(eulerian_graph))

    # ---- Step 6: Shortcutting → Hamiltonian Cycle ----
    visited = [False] * n
    tour = []
    for u, v in eulerian_edges:
        if not visited[u]:
            tour.append(u)
            visited[u] = True
        if not visited[v]:
            tour.append(v)
            visited[v] = True

    # Return to start
    tour.append(tour[0])

    metadata = {
        "algorithm": "Christofides",
        "mst_weight": float(mst_weight),
        "num_odd_vertices": len(odd_vertices),
    }
    return tour, metadata


# ============================================================
# 3. Two-Opt Local Search
# ============================================================

def two_opt_tsp(
    points: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 1000,
    verbose: bool = False,
) -> Tuple[List[int], dict]:
    """Improve a TSP tour using the 2-opt local search heuristic.

    The 2-opt algorithm works by iteratively removing two edges from the tour
    and reconnecting the resulting paths in a different way. It accepts the
    swap if it reduces the total tour length.

    A swap replaces edges (i, i+1) and (j, j+1) with (i, j) and (i+1, j+1),
    which reverses the segment between i+1 and j.

    Complexity: O(n²) per iteration, up to max_iterations iterations

    Args:
        points: numpy array of shape (n, 2)
        initial_tour: starting tour (if None, uses Nearest Neighbor)
        max_iterations: maximum number of improvement iterations
        verbose: whether to print improvement info

    Returns:
        (tour, metadata) where tour is an improved tour
    """
    n = len(points)
    dist_mat = compute_distance_matrix(points)

    # Initialize tour
    if initial_tour is None:
        tour, _ = nearest_neighbor_tsp(points)
        # Remove the final return-to-start for internal representation
        tour = tour[:-1]
    else:
        tour = list(initial_tour)
        if tour[0] == tour[-1]:
            tour = tour[:-1]  # Remove closing vertex for internal ops

    iteration = 0
    total_improvement = 0.0

    while iteration < max_iterations:
        best_improvement = 0.0
        best_i = -1
        best_j = -1

        # Evaluate all possible 2-opt swaps
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip adjacent edges (would not change tour)
                if j == i + 1:
                    continue

                # Current edges: (i, i+1) and (j, j+1)
                # Proposed edges: (i, j) and (i+1, j+1)
                # Note: j+1 wraps around to 0 for the last vertex
                current_cost = (
                    dist_mat[tour[i], tour[i + 1]]
                    + dist_mat[tour[j], tour[(j + 1) % n]]
                )
                new_cost = (
                    dist_mat[tour[i], tour[j]]
                    + dist_mat[tour[i + 1], tour[(j + 1) % n]]
                )
                improvement = current_cost - new_cost

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_i = i
                    best_j = j

        if best_improvement > 1e-10:
            # Apply the best found swap: reverse the segment between i+1 and j
            i = best_i
            j = best_j
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            total_improvement += best_improvement
            iteration += 1
            if verbose and iteration % 100 == 0:
                print(f"  2-opt iter {iteration}: improvement = {best_improvement:.4f}")
        else:
            # No improving move found → local optimum
            break

    # Add return to start
    tour = tour + [tour[0]]

    metadata = {
        "algorithm": "2-opt",
        "iterations": iteration,
        "total_improvement": total_improvement,
    }
    return tour, metadata


def two_opt_fast(
    points: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 1000,
) -> Tuple[List[int], dict]:
    """Vectorized 2-opt using numpy for speed. GPU-friendly version.

    This is inspired by DIFUSCO's batched_two_opt_torch but implemented in
    numpy. It's faster than the pure Python version for n > 100.

    Args:
        points: numpy array of shape (n, 2)
        initial_tour: starting tour
        max_iterations: maximum iterations

    Returns:
        (tour, metadata)
    """
    n = len(points)
    points = points.astype(np.float64)

    if initial_tour is None:
        tour_np = np.arange(n)
        # Simple greedy initialization
        for i in range(1, n):
            remaining = np.setdiff1d(np.arange(n), tour_np[:i])
            dists = np.linalg.norm(points[remaining] - points[tour_np[i - 1]], axis=1)
            tour_np[i] = remaining[np.argmin(dists)]
    else:
        tour_np = np.array(initial_tour)
        if tour_np[0] == tour_np[-1]:
            tour_np = tour_np[:-1]

    for iteration in range(max_iterations):
        # Precompute all edge lengths
        points_i = points[tour_np]          # (n, 2)
        points_i_plus_1 = points[tour_np]   # (n, 2)

        # Roll to get next vertex
        points_j = points[np.roll(tour_np, -1)]  # (n, 2)

        # Broadcast for all pairs
        # For each pair (i, j), compute delta = d(i,j) + d(i+1,j+1) - d(i,i+1) - d(j,j+1)
        pi = points_i[:, None, :]     # (n, 1, 2)
        pj = points_i[None, :, :]     # (1, n, 2)
        pi1 = points_j[:, None, :]    # (n, 1, 2)
        pj1 = points_j[None, :, :]    # (1, n, 2)

        A_ij = np.sqrt(np.sum((pi - pj) ** 2, axis=-1))          # (n, n)
        A_i1_j1 = np.sqrt(np.sum((pi1 - pj1) ** 2, axis=-1))    # (n, n)
        A_i_i1 = np.sqrt(np.sum((pi - pi1) ** 2, axis=-1))       # (n, n)
        A_j_j1 = np.sqrt(np.sum((pj - pj1) ** 2, axis=-1))       # (n, n)

        change = A_ij + A_i1_j1 - A_i_i1 - A_j_j1

        # Only consider i < j-1 (non-adjacent)
        mask = np.triu(np.ones((n, n)), k=2)
        change = change * mask

        # Find best swap
        min_idx = np.argmin(change)
        min_i = min_idx // n
        min_j = min_idx % n
        min_val = change[min_i, min_j]

        if min_val < -1e-10:
            # Reverse segment between min_i+1 and min_j
            tour_np[min_i + 1 : min_j + 1] = np.flip(tour_np[min_i + 1 : min_j + 1])
        else:
            break

    tour = tour_np.tolist() + [tour_np[0]]
    metadata = {
        "algorithm": "2-opt (numpy)",
        "iterations": iteration + 1,
    }
    return tour, metadata


# ============================================================
# 4. Combined Pipeline: Christofides + 2-opt
# ============================================================

def christofides_with_2opt(
    points: np.ndarray,
    max_2opt_iterations: int = 1000,
    use_fast_2opt: bool = True,
) -> Tuple[List[int], dict]:
    """Run Christofides algorithm followed by 2-opt refinement.

    This is the recommended baseline for comparing with DIFUSCO,
    as DIFUSCO uses 2-opt as a post-processing step.

    Args:
        points: numpy array of shape (n, 2)
        max_2opt_iterations: max iterations for 2-opt
        use_fast_2opt: use numpy-vectorized 2-opt for speed

    Returns:
        (tour, metadata)
    """
    # Step 1: Christofides
    christofides_tour, christofides_meta = christofides_tsp(points)
    christofides_cost = tour_cost(compute_distance_matrix(points), christofides_tour)

    # Step 2: 2-opt refinement
    if use_fast_2opt:
        refined_tour, two_opt_meta = two_opt_fast(
            points, christofides_tour, max_iterations=max_2opt_iterations
        )
    else:
        refined_tour, two_opt_meta = two_opt_tsp(
            points, christofides_tour, max_iterations=max_2opt_iterations
        )
    refined_cost = tour_cost(compute_distance_matrix(points), refined_tour)

    metadata = {
        "algorithm": "Christofides+2opt",
        "christofides_cost": christofides_cost,
        "refined_cost": refined_cost,
        "improvement_pct": 100 * (christofides_cost - refined_cost) / christofides_cost,
        "mst_weight": christofides_meta.get("mst_weight"),
        "two_opt_iterations": two_opt_meta["iterations"],
    }
    return refined_tour, metadata
