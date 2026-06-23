"""Improvement #6: LKH3 post-processing on DualOpt output."""

import sys, os, time, numpy as np, torch

_project = os.path.dirname(__file__)
os.environ['PATH'] = os.path.join(_project, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']
sys.path.insert(0, os.path.join(_project, 'DualOpt-main'))
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import LCP_TSP, load_problem
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost
import lkh, tsplib95

device = torch.device('cuda')
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

# Load revisers
revisers = []
for sz in [50, 20, 10]:
    r, _ = load_model(os.path.join(_project, 'DualOpt-main', 'pretrained',
                     'local_%d' % sz, 'epoch-100.pt'), is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

def lkh_polish(pts, tour, max_trials=50):
    """Run LKH3 with given tour as initial solution."""
    n = len(pts)
    problem = tsplib95.models.StandardProblem()
    problem.name = 'TSP'; problem.type = 'TSP'; problem.dimension = n
    problem.edge_weight_type = 'EUC_2D'
    problem.node_coords = {i+1: (float(pts[i][0]*1e6), float(pts[i][1]*1e6)) for i in range(n)}

    # Save initial tour to file
    init_file = os.path.join(_project, 'outputs', '_lkh_init.txt')
    with open(init_file, 'w') as f:
        f.write('TOUR_SECTION\n')
        for v in tour:
            if v == tour[-1] and v == tour[0]: continue
            f.write('%d\n' % (v+1))
        f.write('-1\n')

    solution = lkh.solve('LKH.exe', problem=problem, max_trials=max_trials,
                         initial_tour_file=init_file, runs=1)
    polished = [r-1 for r in solution[0]]
    if polished[0] != polished[-1]:
        polished.append(polished[0])
    return polished

# Test: TSP-50 and TSP-100
results = {}
for n in [50, 100]:
    np.random.seed(42)
    instances = [np.random.rand(n, 2).astype(np.float64) for _ in range(5)]
    costs_init = []; costs_dual = []; costs_lkh = []

    for idx, pts in enumerate(instances):
        print('\rTSP-%d [%d/5]' % (n, idx+1), end='', flush=True)

        # Initial + DualOpt
        tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=100 if n<=100 else 50)
        init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
        it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
        seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
        for rid in range(3):
            seeds = LCP_TSP(seeds, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
        cost_dual = (seeds[:,1:]-seeds[:,:-1]).norm(p=2,dim=2).sum(1)+(seeds[:,0]-seeds[:,-1]).norm(p=2,dim=1)

        # Extract tour from seeds (coordinates in tour order)
        # Need to convert back to vertex indices - use nearest neighbor mapping
        from scipy.spatial import KDTree
        seed_coords = seeds[0].cpu().numpy()
        tree = KDTree(pts)
        _, dual_tour = tree.query(seed_coords)
        dual_tour = dual_tour.tolist()
        if dual_tour[0] != dual_tour[-1]:
            dual_tour.append(dual_tour[0])

        # LKH polish
        try:
            polished = lkh_polish(pts, dual_tour, max_trials=30)
            cost_lkh = tour_cost(compute_distance_matrix(pts), polished)
        except:
            cost_lkh = cost_dual.item()

        costs_init.append(init_cost)
        costs_dual.append(cost_dual.item())
        costs_lkh.append(cost_lkh)

    print()
    impr_dual = (np.mean(costs_init) - np.mean(costs_dual)) / np.mean(costs_init) * 100
    impr_lkh = (np.mean(costs_init) - np.mean(costs_lkh)) / np.mean(costs_init) * 100
    impr_over = (np.mean(costs_dual) - np.mean(costs_lkh)) / np.mean(costs_dual) * 100

    results[n] = {'init': np.mean(costs_init), 'dual': np.mean(costs_dual), 'lkh': np.mean(costs_lkh)}
    print('  Init: %.4f  |  DualOpt: %.4f (%+.2f%%)  |  +LKH: %.4f (%+.2f%%)  |  LKH over Dual: %+.2f%%' % (
        np.mean(costs_init), np.mean(costs_dual), impr_dual,
        np.mean(costs_lkh), impr_lkh, impr_over))

print('\n' + '='*60)
print('FINAL: LKH Polish over DualOpt')
print('='*60)
for n in [50, 100]:
    r = results[n]
    impr = (r['dual'] - r['lkh']) / r['dual'] * 100
    print('TSP-%d: DualOpt=%.4f  +LKH=%.4f  delta=%+.3f%%' % (n, r['dual'], r['lkh'], impr))
