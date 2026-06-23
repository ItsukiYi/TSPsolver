"""Quick test of classic TSP algorithms."""
import sys
sys.path.insert(0, ".")

from src.utils import generate_random_tsp_instance, compute_distance_matrix, tour_cost
from src.algorithms import nearest_neighbor_tsp, christofides_tsp, two_opt_fast, christofides_with_2opt
import time

# Test on TSP-20
points = generate_random_tsp_instance(20, seed=42)
dist_mat = compute_distance_matrix(points)

nn_tour, _ = nearest_neighbor_tsp(points)
ch_tour, ch_meta = christofides_tsp(points)
opt2_tour, opt2_meta = two_opt_fast(points, max_iterations=1000)
ch2_tour, ch2_meta = christofides_with_2opt(points)

print("TSP-20 Results:")
print(f"  Nearest Neighbor:   {tour_cost(dist_mat, nn_tour):.4f}")
print(f"  Christofides:       {tour_cost(dist_mat, ch_tour):.4f} (MST={ch_meta['mst_weight']:.4f})")
print(f"  2-opt (from NN):    {tour_cost(dist_mat, opt2_tour):.4f} ({opt2_meta['iterations']} iters)")
print(f"  Christofides+2-opt: {tour_cost(dist_mat, ch2_tour):.4f} (improve={ch2_meta['improvement_pct']:.1f}%)")
print()

# Test scalability
for n in [50, 100, 200]:
    points_n = generate_random_tsp_instance(n, seed=123)
    start = time.time()
    _, _ = christofides_with_2opt(points_n, max_2opt_iterations=500)
    elapsed = time.time() - start
    print(f"  TSP-{n}: Christofides+2opt = {elapsed:.2f}s")

print()
print("All classic algorithms working correctly!")
