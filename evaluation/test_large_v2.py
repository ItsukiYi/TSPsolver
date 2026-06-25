"""Large-scale test: all DIFUSCO ops in subprocess, DualOpt in main."""

import sys, os, time, json, math, subprocess, numpy as np, torch, pickle

_project = os.path.dirname(__file__)
_venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')

# ---- Step 1: Generate heatmaps via subprocess ----
np.random.seed(42)
sizes = [100, 200]
all_pts = {str(n): [np.random.rand(n, 2).astype(np.float64) for _ in range(3)] for n in sizes}

with open(os.path.join(_project, 'outputs', '_pts.pkl'), 'wb') as f:
    pickle.dump(all_pts, f)

print('Step 1: Generating heatmaps...')
subprocess.run([_venv_py, '-c', r'''
import sys, os, pickle, numpy as np, torch
_p = r"''' + _project + r'''"
sys.path.insert(0, os.path.join(_p, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_p, 'DIFUSCO-main'))
from pl_tsp_model import TSPModel
from argparse import Namespace
from utils.diffusion_schedulers import InferenceSchedule

args = Namespace(diffusion_type='categorical', diffusion_schedule='cosine',
    diffusion_steps=1000, inference_diffusion_steps=50,
    inference_schedule='cosine', inference_trick='ddim',
    n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation='sum',
    two_opt_iterations=0, parallel_sampling=1, sequential_sampling=1,
    save_numpy_heatmap=False, storage_path='.',
    training_split='data/tsp_problems/tsp50_test.txt',
    validation_split='data/tsp_problems/tsp50_test.txt',
    test_split='data/tsp_problems/tsp50_test.txt',
    batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
    lr_scheduler='cosine-decay', num_epochs=50, num_workers=0,
    validation_examples=8, use_activation_checkpoint=False, fp16=False)
device = torch.device('cuda')
ckpt = os.path.join(_p, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

with open(os.path.join(_p, 'outputs', '_pts.pkl'), 'rb') as f:
    all_pts = pickle.load(f)

all_hm = {}
for size_str, instances in all_pts.items():
    hms = []
    for pts in instances:
        n = len(pts)
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
            ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
            for i in range(50):
                t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
                xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
            hms.append(xt.float().cpu().numpy().squeeze() + 1e-6)
    all_hm[size_str] = hms
    print('Heatmaps done for TSP-%s' % size_str)

with open(os.path.join(_p, 'outputs', '_hm.pkl'), 'wb') as f:
    pickle.dump(all_hm, f)
print('All heatmaps saved')
'''], check=True, timeout=600)
print('Heatmaps done.\n')

# ---- Step 2: Get DIFUSCO tours via subprocess ----
print('Step 2: DIFUSCO greedy merge...')
subprocess.run([_venv_py, '-c', r'''
import sys, os, pickle, numpy as np
_p = r"''' + _project + r'''"
sys.path.insert(0, os.path.join(_p, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_p, 'DIFUSCO-main'))
from utils.tsp_utils import merge_tours

with open(os.path.join(_p, 'outputs', '_hm.pkl'), 'rb') as f:
    all_hm = pickle.load(f)
with open(os.path.join(_p, 'outputs', '_pts.pkl'), 'rb') as f:
    all_pts = pickle.load(f)

all_tours = {}
for size_str in all_pts:
    tours_list = []
    for idx, pts in enumerate(all_pts[size_str]):
        hm = all_hm[size_str][idx]
        np_pts = pts.astype(np.float64)
        tours, _ = merge_tours(hm[np.newaxis,:,:], np_pts, None, sparse_graph=False, parallel_sampling=1)
        tours_list.append(tours[0])
    all_tours[size_str] = tours_list
    print('Tours done for TSP-%s' % size_str)

with open(os.path.join(_p, 'outputs', '_tours.pkl'), 'wb') as f:
    pickle.dump(all_tours, f)
print('All tours saved')
'''], check=True, timeout=60)
print('Tours done.\n')

# ---- Step 3: Run DualOpt methods ----
print('Step 3: Running methods...')
sys.path.insert(0, os.path.join(_project, 'DualOpt-improved'))
sys.path.insert(1, os.path.join(_project, 'DualOpt-main'))
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import LCP_TSP, load_problem
from utils.destroy_repair import destroy_repair_cycle, compute_edge_confidence, destroy, repair_greedy, repair_2opt
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost
from scipy.spatial.distance import pdist, squareform

with open(os.path.join(_project, 'outputs', '_pts.pkl'), 'rb') as f:
    all_pts = pickle.load(f)
with open(os.path.join(_project, 'outputs', '_hm.pkl'), 'rb') as f:
    all_hm = pickle.load(f)
with open(os.path.join(_project, 'outputs', '_tours.pkl'), 'rb') as f:
    all_tours = pickle.load(f)

device = torch.device('cuda')
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

results = {}

for size_str in ['100', '200']:
    n = int(size_str)
    results[size_str] = {'orig': [], 'pipe': [], 'dr': []}
    print('\n' + '='*50)
    print('TSP-%d' % n)

    revisers = []
    for sz in [50, 20, 10]:
        path = os.path.join(_project, 'DualOpt-main', 'pretrained',
                           'local_%d' % sz, 'epoch-100.pt')
        r, _ = load_model(path, is_local=True)
        r.to(device); r.eval(); r.set_decode_type('greedy')
        revisers.append(r)

    for idx in range(len(all_pts[size_str])):
        pts = np.array(all_pts[size_str][idx], dtype=np.float64)
        hm = np.array(all_hm[size_str][idx], dtype=np.float64)
        dif_tour = all_tours[size_str][idx]

        print('\n[%d/%d]' % (idx+1, len(all_pts[size_str])))

        # Initial
        max_2opt = 100 if n <= 200 else 30
        tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=max_2opt)
        init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
        it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
        seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
        print('  C+2opt init: %.2f' % init_cost)

        # Original DualOpt
        t0 = time.time()
        s = seeds.clone()
        for rid in range(3):
            s = LCP_TSP(s, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
        cost_orig = (s[:,1:]-s[:,:-1]).norm(p=2,dim=2).sum(1)+(s[:,0]-s[:,-1]).norm(p=2,dim=1)
        impr_orig = (init_cost - cost_orig.item()) / init_cost * 100
        print('  Original:     %.2f (%+.2f%%)' % (cost_orig.item(), impr_orig))
        results[size_str]['orig'].append(impr_orig)

        # #2 DIFUSCO -> DualOpt
        if dif_tour[-1] == dif_tour[0]: dif_tour = dif_tour[:-1]
        seeds_d = torch.from_numpy(pts[dif_tour]).float().unsqueeze(0).to(device)
        s_d = seeds_d.clone()
        for rid in range(3):
            s_d = LCP_TSP(s_d, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
        cost_pipe = (s_d[:,1:]-s_d[:,:-1]).norm(p=2,dim=2).sum(1)+(s_d[:,0]-s_d[:,-1]).norm(p=2,dim=1)
        impr_pipe = (init_cost - cost_pipe.item()) / init_cost * 100
        print('  #2 Pipeline:  %.2f (%+.2f%%)' % (cost_pipe.item(), impr_pipe))
        results[size_str]['pipe'].append(impr_pipe)

        # #5 Destroy-and-Repair (greedy+2opt, best of K=3,5)
        confs = compute_edge_confidence(hm, it)
        dists = squareform(pdist(pts))
        best_dr = init_cost
        for K in [3, 5]:
            segments, _ = destroy(it, confs, K)
            if len(segments) <= 1: continue
            repaired, _ = repair_greedy(segments, pts)
            repaired = repair_2opt(repaired, pts, max_iter=200)
            c = sum(dists[repaired[i], repaired[(i+1)%n]] for i in range(n))
            if c < best_dr: best_dr = c
        impr_dr = (init_cost - best_dr) / init_cost * 100
        print('  #5 Dst-Repair: %.2f (%+.2f%%)' % (best_dr, impr_dr))
        results[size_str]['dr'].append(impr_dr)

# Summary
print('\n' + '='*60)
print('FINAL SUMMARY')
print('='*60)
for size_str in ['100', '200']:
    r = results[size_str]
    print('TSP-%s:' % size_str)
    print('  Original DualOpt:   %+.2f%%' % np.mean(r['orig']))
    print('  #2 DIFUSCO->DualOpt: %+.2f%%' % np.mean(r['pipe']))
    print('  #5 Destroy-Repair:   %+.2f%%' % np.mean(r['dr']))

with open(os.path.join(_project, 'outputs', '_large_results.json'), 'w') as f:
    json.dump({k: {kk: [float(vv) for vv in v] for kk, v in r.items()}
               for k, r in results.items()}, f, indent=2)
print('\nSaved.')
