"""Test hyper repair with pre-saved heatmaps."""
import sys, os, pickle, numpy as np, torch

_project = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_project, 'DualOpt-improved'))
sys.path.insert(1, os.path.join(_project, 'DualOpt-main'))
sys.path.append(os.path.join(_project, 'src'))

with open(os.path.join(_project, 'outputs', '_test_heatmaps.pkl'), 'rb') as f:
    data = pickle.load(f)
instances = data['instances']
heatmaps = data['heatmaps']

from utils import load_model
from utils.functions import load_problem
from utils.hyper_repair import hyper_repair_with_dualopt
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

device = torch.device('cuda')
revisers = []
for size in [50, 20, 10]:
    path = os.path.join(_project, 'DualOpt-improved', 'pretrained',
                        'local_{}'.format(size), 'epoch-100.pt')
    r, _ = load_model(path, is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

class O:
    revision_lens = [50, 20, 10]
    revision_iters = [25, 10, 5]
    problem = 'tsp'
    lkh_layer_number = 2

opts = O()

print('Improvement #5b: Hyper-Edge Repair')
for idx, pts in enumerate(instances):
    print('[%d/3]' % (idx+1))
    hm = heatmaps[idx]
    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=100)
    init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
    it = tour_c2[:-1] if tour_c2[-1] == tour_c2[0] else tour_c2
    for K in [3, 5, 7]:
        try:
            repaired, cost, improved = hyper_repair_with_dualopt(pts, it, hm, K, revisers, opts)
            delta = (init_cost - cost) / init_cost * 100
            s = 'IMPROVED' if improved else 'SAME'
            print('  K=%d: %.4f -> %.4f (%+.2f%%) [%s]' % (K, init_cost, cost, delta, s))
        except Exception as e:
            print('  K=%d: FAILED - %s' % (K, str(e)[:120]))
