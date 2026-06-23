"""LKH baseline on TSPLIB instances."""
import sys, os, numpy as np, time, json

_project = os.path.dirname(__file__)

# Setup LKH
os.environ['PATH'] = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']

import lkh
import tsplib95
sys.path.insert(0, os.path.join(_project, 'src'))
from src.tsplib_loader import load_tsplib_instance
from src.utils import compute_distance_matrix, tour_cost

# Load all TSPLIB instances
instance_names = ['eil51', 'berlin52', 'eil76', 'kroA100', 'ch150', 'tsp225', 'a280', 'pr1002']
print('=' * 60)
print('LKH3 BASELINE ON TSPLIB')
print('=' * 60)
print(f'{"Instance":<12s} {"n":>5s} {"LKH Cost":>12s} {"Known OPT":>12s} {"Gap":>8s} {"Time":>8s}')
print('-' * 60)

results = {}
for name in instance_names:
    pts, _, opt_cost, display = load_tsplib_instance(name)
    n = len(pts)

    # Build TSPLIB problem
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'
    problem.type = 'TSP'
    problem.dimension = n
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {i+1: (float(pts[i][0]), float(pts[i][1])) for i in range(n)}

    # Run LKH
    t0 = time.time()
    try:
        solution = lkh.solve('LKH.exe', problem=problem, max_trials=n, runs=10)
        tour = [r - 1 for r in solution[0]]  # 1-indexed -> 0-indexed
        if tour[0] != tour[-1]:
            tour.append(tour[0])
        cost = tour_cost(compute_distance_matrix(pts), tour)
        elapsed = time.time() - t0
        gap = (cost/(opt_cost or 1) - 1)*100 if opt_cost else None
        gap_str = f'{gap:.2f}%' if gap is not None else 'N/A'
        print(f'{name:<12s} {n:5d} {cost:12.2f} {opt_cost or 0:12.0f} {gap_str:>8s} {elapsed:7.1f}s')
        results[name] = {'cost': float(cost), 'time': elapsed, 'gap': gap, 'opt': opt_cost}
    except Exception as e:
        print(f'{name:<12s} {n:5d} FAILED: {str(e)[:80]}')
        results[name] = {'error': str(e)[:200]}

with open('outputs/lkh_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to outputs/lkh_baseline.json')
