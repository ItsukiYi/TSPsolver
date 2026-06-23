"""Scalability benchmark: Classic + LKH across TSP sizes."""

import sys, os, numpy as np, time, json, math
from datetime import datetime

_project = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_project, 'src'))

# LKH setup
os.environ['PATH'] = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']

from src.utils import generate_random_tsp_instance, compute_distance_matrix, tour_cost
from src.algorithms import nearest_neighbor_tsp, christofides_tsp, christofides_with_2opt
import lkh, tsplib95

SIZES = [100, 200, 500, 1000]
TRIALS = 10
SEED = 42

print('=' * 70)
print(f'SCALABILITY BENCHMARK: TSP {SIZES} ({TRIALS} instances each)')
print('=' * 70)

all_results = {}

for n in SIZES:
    print(f'\n--- TSP-{n} ---')
    np.random.seed(SEED)
    instances = [generate_random_tsp_instance(n, seed=SEED + i) for i in range(TRIALS)]

    results = {}

    # NN
    t0 = time.time()
    costs = []
    for pts in instances:
        tour, _ = nearest_neighbor_tsp(pts)
        costs.append(tour_cost(compute_distance_matrix(pts), tour))
    t = time.time() - t0
    results['NN'] = {'mean': np.mean(costs), 'std': np.std(costs), 'time': t/TRIALS}
    print(f'  NN:              {results["NN"]["mean"]:.2f} +- {results["NN"]["std"]:.2f}  ({results["NN"]["time"]*1000:.1f}ms)')

    # Christofides
    t0 = time.time()
    costs = []
    for pts in instances:
        tour, _ = christofides_tsp(pts)
        costs.append(tour_cost(compute_distance_matrix(pts), tour))
    t = time.time() - t0
    results['Christofides'] = {'mean': np.mean(costs), 'std': np.std(costs), 'time': t/TRIALS}
    gap = (results['Christofides']['mean'] / results['NN']['mean'] - 1) * 100
    print(f'  Christofides:    {results["Christofides"]["mean"]:.2f} +- {results["Christofides"]["std"]:.2f}  ({results["Christofides"]["time"]:.2f}s, {gap:+.1f}% vs NN)')

    # C+2opt
    t0 = time.time()
    costs = []
    for pts in instances:
        tour, _ = christofides_with_2opt(pts, max_2opt_iterations=1000)
        costs.append(tour_cost(compute_distance_matrix(pts), tour))
    t = time.time() - t0
    results['C+2opt'] = {'mean': np.mean(costs), 'std': np.std(costs), 'time': t/TRIALS}
    gap = (results['C+2opt']['mean'] / results['NN']['mean'] - 1) * 100
    print(f'  C+2opt:          {results["C+2opt"]["mean"]:.2f} +- {results["C+2opt"]["std"]:.2f}  ({results["C+2opt"]["time"]:.2f}s, {gap:+.1f}% vs NN)')

    # LKH (scale coords by 1e6 for TSPLIB integer precision)
    SCALE = 1000000
    t0 = time.time()
    costs = []
    for pts in instances:
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'; problem.type = 'TSP'; problem.dimension = n
        problem.edge_weight_type = 'EUC_2D'
        problem.node_coords = {i+1: (int(pts[i][0]*SCALE), int(pts[i][1]*SCALE)) for i in range(n)}
        try:
            solution = lkh.solve('LKH.exe', problem=problem, max_trials=min(n, 100), runs=10)
            tour = [r - 1 for r in solution[0]]
            if tour[0] != tour[-1]: tour.append(tour[0])
            # Cost computed on ORIGINAL coordinates (not scaled)
            costs.append(tour_cost(compute_distance_matrix(pts), tour))
        except Exception as e:
            print(f'  LKH error: {e}')
            costs.append(float('nan'))
    t = time.time() - t0
    valid_costs = [c for c in costs if not math.isnan(c)]
    if valid_costs:
        results['LKH'] = {'mean': np.mean(valid_costs), 'std': np.std(valid_costs), 'time': t/len(instances)}
        gap = (results['LKH']['mean'] / results['NN']['mean'] - 1) * 100
        print(f'  LKH:             {results["LKH"]["mean"]:.2f} +- {results["LKH"]["std"]:.2f}  ({results["LKH"]["time"]:.2f}s, {gap:+.1f}% vs NN)')
    else:
        results['LKH'] = {'error': 'all failed'}

    all_results[f'TSP-{n}'] = results

# Summary table
print(f'\n{"=" * 70}')
print(f'SUMMARY: Cost vs Size')
print(f'{"=" * 70}')
header = f'{"Size":>8s}'
for m in ['NN', 'Christofides', 'C+2opt', 'LKH']:
    header += f'  {m:>12s}'
print(header)
print('-' * 62)

opt_est = {100: 7.76, 200: 10.7, 500: 16.5, 1000: 23.1}  # ~0.712*sqrt(n)
for n in SIZES:
    row = f'{n:8d}'
    r = all_results[f'TSP-{n}']
    for m in ['NN', 'Christofides', 'C+2opt', 'LKH']:
        if m in r and 'mean' in r[m]:
            row += f'  {r[m]["mean"]:12.4f}'
        else:
            row += f'  {"N/A":>12s}'
    print(row)

# Runtime table
print(f'\n{"=" * 70}')
print(f'SUMMARY: Runtime per Instance (seconds)')
print(f'{"=" * 70}')
header = f'{"Size":>8s}'
for m in ['NN', 'Christofides', 'C+2opt', 'LKH']:
    header += f'  {m:>10s}'
print(header)
print('-' * 52)
for n in SIZES:
    row = f'{n:8d}'
    r = all_results[f'TSP-{n}']
    for m in ['NN', 'Christofides', 'C+2opt', 'LKH']:
        if m in r and 'mean' in r[m]:
            t = r[m]['time']
            if t < 1:
                row += f'  {t*1000:8.1f}ms'
            else:
                row += f'  {t:8.2f}s'
        else:
            row += f'  {"N/A":>10s}'
    print(row)

# Gap vs LKH
print(f'\n{"=" * 70}')
print(f'SUMMARY: Gap vs LKH (%)')
print(f'{"=" * 70}')
header = f'{"Size":>8s}'
for m in ['NN', 'Christofides', 'C+2opt']:
    header += f'  {m:>10s}'
print(header)
print('-' * 42)
for n in SIZES:
    r = all_results[f'TSP-{n}']
    if 'LKH' in r and 'mean' in r['LKH']:
        lkh_ref = r['LKH']['mean']
        row = f'{n:8d}'
        for m in ['NN', 'Christofides', 'C+2opt']:
            gap = (r[m]['mean'] / lkh_ref - 1) * 100
            row += f'  {gap:+9.1f}%'
        print(row)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Convert to serializable
out = {}
for k, v in all_results.items():
    out[k] = {m: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                  for kk, vv in d.items()} for m, d in v.items()}
with open(f'outputs/scalability_{timestamp}.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'\nSaved to outputs/scalability_{timestamp}.json')
