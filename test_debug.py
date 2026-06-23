"""Debug Christofides algorithm."""
import sys
sys.path.insert(0, ".")

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.optimize import linear_sum_assignment
import networkx as nx
from src.utils import generate_random_tsp_instance, compute_distance_matrix

points = generate_random_tsp_instance(20, seed=42)
dist_mat = compute_distance_matrix(points)
n = len(points)

# Step 1: MST
mst_sparse = minimum_spanning_tree(dist_mat)
mst_graph = nx.from_scipy_sparse_array(mst_sparse)
print(f"MST edges: {mst_graph.number_of_edges()}")

# Step 2: Odd vertices
odd_vertices = [v for v in range(n) if mst_graph.degree(v) % 2 == 1]
print(f"Odd vertices ({len(odd_vertices)}): {odd_vertices}")
print(f"MST degrees: {[mst_graph.degree(v) for v in range(n)]}")

# Step 3: MWPM
odd_dist_mat = dist_mat[np.ix_(odd_vertices, odd_vertices)].copy()
np.fill_diagonal(odd_dist_mat, 1e12)
row_ind, col_ind = linear_sum_assignment(odd_dist_mat)
print(f"Assignment: rows={row_ind}, cols={col_ind}")

# Build matching
matching_edges = []
seen_pairs = set()
for i, j in zip(row_ind, col_ind):
    u = odd_vertices[i]
    v = odd_vertices[j]
    if u == v:
        print(f"  SKIP self-match: {u}")
        continue
    if u > v:
        u, v = v, u
    if (u, v) in seen_pairs:
        print(f"  SKIP duplicate: ({u},{v})")
        continue
    seen_pairs.add((u, v))
    matching_edges.append((u, v))
    print(f"  Match: {u} <-> {v}")

print(f"Matching edges: {matching_edges}")

# Step 4: Eulerian graph
eulerian_graph = nx.MultiGraph()
for u, v in mst_graph.edges():
    eulerian_graph.add_edge(u, v)
for u, v in matching_edges:
    eulerian_graph.add_edge(u, v)

print(f"\nEulerian graph:")
print(f"  Nodes: {eulerian_graph.number_of_nodes()}")
print(f"  Edges: {eulerian_graph.number_of_edges()}")
print(f"  Is connected: {nx.is_connected(eulerian_graph)}")
degrees = [eulerian_graph.degree(v) for v in range(n)]
print(f"  Degrees: {degrees}")
print(f"  Odd degrees: {[v for v in range(n) if degrees[v] % 2 == 1]}")
print(f"  Is Eulerian: {nx.is_eulerian(eulerian_graph)}")

if not nx.is_eulerian(eulerian_graph):
    print("\nTrying to fix by adding extra edges between unmatched odd vertices...")
    # Check which vertices still have odd degree
    odd_now = [v for v in range(n) if eulerian_graph.degree(v) % 2 == 1]
    print(f"  Still odd: {odd_now}")
    # Pair them up greedily
    for i in range(0, len(odd_now), 2):
        if i + 1 < len(odd_now):
            u, v = odd_now[i], odd_now[i+1]
            eulerian_graph.add_edge(u, v)
            print(f"  Added edge to fix: {u} <-> {v}")
    print(f"  Now Eulerian: {nx.is_eulerian(eulerian_graph)}")
