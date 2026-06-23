r"""Test improvements on TSP-100/200 — using sliding-window DualOpt reviser.

Key insight: DualOpt reviser (LCP_TSP) works on any size via sliding windows.
Only the grid divide-and-conquer first step fails on n>100.
"""

import sys, os, time, json, math, pickle, subprocess
import numpy as np

_project = os.path.dirname(__file__)
_venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')

# ---- Phase 1: Generate instances + DIFUSCO heatmaps (subprocess) ----
gen_script = r"""
import sys, os, numpy as np, torch, pickle, json
_project = r'{project}'
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
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
ckpt = os.path.join(_project, 'tsp50_categorical/checkpoints/epoch=6-step=105.ckpt')
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

np.random.seed(42)
sizes = [100, 200]
all_data = {{}}

for n in sizes:
    instances = [np.random.rand(n, 2).astype(np.float64) for _ in range(3)]
    heatmaps = []
    for pts in instances:
        with torch.no_grad():
            pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
            xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
            ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
            for i in range(50):
                t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
                xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
            heatmaps.append(xt.float().cpu().numpy().squeeze() + 1e-6)
    all_data[str(n)] = {{'instances': [p.tolist() for p in instances],
                          'heatmaps': [h.tolist() for h in heatmaps]}}
    print('Generated TSP-%d heatmaps' % n)

with open(r'{data_path}', 'w') as f:
    json.dump(all_data, f)
print('Data saved')
""".format(project=_project, data_path=os.path.join(_project, 'outputs', '_large_data.json'))

print('Generating heatmaps (subprocess)...')
subprocess.run([_venv_py, '-c', gen_script], check=True, timeout=600)
print('Done.\n')

# ---- Phase 2: Run all methods (separate subprocess) ----
test_script = r"""
import sys, os, time, json, math, numpy as np, torch
_project = r'{project}'
sys.path.insert(0, os.path.join(_project, 'DualOpt-improved'))
sys.path.insert(1, os.path.join(_project, 'DualOpt-main'))
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import LCP_TSP, load_problem
from utils.destroy_repair import destroy_repair_cycle
from utils.heatmap_guide import heatmap_guided_LCP_TSP as hg
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost
from scipy.spatial.distance import pdist, squareform

with open(r'{data_path}') as f:
    all_data = json.load(f)

device = torch.device('cuda')
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

results = {{}}

for size_str in ['100', '200']:
    n = int(size_str)
    data = all_data[size_str]
    results[size_str] = {{}}

    for idx in range(len(data['instances'])):
        pts = np.array(data['instances'][idx], dtype=np.float64)
        hm = np.array(data['heatmaps'][idx], dtype=np.float64)

        print('\nTSP-%d [%d/%d]' % (n, idx+1, len(data['instances'])))

        # Initial tour
        max_2opt = 100 if n <= 200 else 30
        tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=max_2opt)
        init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
        it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
        seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
        print('  Init: %.2f' % init_cost)

        # Load revisers (once per size, but simpler per-instance)
        revisers = []
        for size in [50, 20, 10]:
            path = os.path.join(_project, 'DualOpt-main', 'pretrained', 'local_'+str(size), 'epoch-100.pt')
            r, _ = load_model(path, is_local=True)
            r.to(device); r.eval(); r.set_decode_type('greedy')
            revisers.append(r)

        # Method A: Original DualOpt (sliding window reviser only)
        t0 = time.time()
        s = seeds.clone()
        for rid in range(len(revisers)):
            s = LCP_TSP(s, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
        cost_orig = (s[:,1:]-s[:,:-1]).norm(p=2,dim=2).sum(1)+(s[:,0]-s[:,-1]).norm(p=2,dim=1)
        t_orig = time.time() - t0
        impr_orig = (init_cost - cost_orig.item()) / init_cost * 100
        print('  Original:     %.2f (%.1fs, %+.2f%%)' % (cost_orig.item(), t_orig, impr_orig))

        # Method B: #2 DIFUSCO->DualOpt (DIFUSCO tour as initial)
        # Use DIFUSCO's merge_tours via temporary path isolation
        _saved = list(sys.path)
        sys.path = [os.path.join(_project, 'DIFUSCO-main', 'difusco'),
                    os.path.join(_project, 'DIFUSCO-main')] + \
                   [p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
        for k in list(sys.modules.keys()):
            if k.startswith('utils'): del sys.modules[k]
        from utils.tsp_utils import merge_tours
        tours_dif, _ = merge_tours(hm[np.newaxis,:,:], pts, None, sparse_graph=False, parallel_sampling=1)
        sys.path = _saved
        for k in list(sys.modules.keys()):
            if k.startswith('utils'): del sys.modules[k]
        if dif_tour[-1] == dif_tour[0]: dif_tour = dif_tour[:-1]
        seeds_d = torch.from_numpy(pts[dif_tour]).float().unsqueeze(0).to(device)
        t1 = time.time()
        s_d = seeds_d.clone()
        for rid in range(len(revisers)):
            s_d = LCP_TSP(s_d, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
        cost_pipe = (s_d[:,1:]-s_d[:,:-1]).norm(p=2,dim=2).sum(1)+(s_d[:,0]-s_d[:,-1]).norm(p=2,dim=1)
        t_pipe = time.time() - t1
        impr_pipe = (init_cost - cost_pipe.item()) / init_cost * 100
        print('  #2 Pipeline:  %.2f (%.1fs, %+.2f%%)' % (cost_pipe.item(), t_pipe, impr_pipe))

        # Method C: #5 Destroy-and-Repair (no DualOpt polish, use segment enumeration)
        from utils.destroy_repair import compute_edge_confidence, destroy
        confs = compute_edge_confidence(hm, it)
        best_dr_cost = init_cost
        for K in [3, 5]:
            segments, _ = destroy(it, confs, K)
            if len(segments) <= 1: continue
            dists = squareform(pdist(pts))
            # Greedy repair + 2-opt
            from utils.destroy_repair import repair_greedy, repair_2opt
            repaired, _ = repair_greedy(segments, pts)
            repaired = repair_2opt(repaired, pts, max_iter=200)
            cost_dr = sum(dists[repaired[i], repaired[(i+1)%n]] for i in range(n))
            if cost_dr < best_dr_cost:
                best_dr_cost = cost_dr
        impr_dr = (init_cost - best_dr_cost) / init_cost * 100
        print('  #5 Dstr-Repair: %.2f (%+.2f%%)' % (best_dr_cost, impr_dr))

        results[size_str][str(idx)] = {{
            'init': float(init_cost),
            'orig': float(cost_orig.item()), 'orig_impr': float(impr_orig),
            'pipe': float(cost_pipe.item()), 'pipe_impr': float(impr_pipe),
            'dr': float(best_dr_cost), 'dr_impr': float(impr_dr),
        }}

# Summary
print('\n' + '='*60)
for size_str in ['100', '200']:
    r = results[size_str]
    origs = [r[k]['orig_impr'] for k in r]
    pipes = [r[k]['pipe_impr'] for k in r]
    drs = [r[k]['dr_impr'] for k in r]
    print('TSP-%s: orig=%+.2f%%  #2=%+.2f%%  #5=%+.2f%%' % (
        size_str, np.mean(origs), np.mean(pipes), np.mean(drs)))

with open(r'{out_path}', 'w') as f:
    json.dump(results, f)
print('Saved to %s' % r'{out_path}')
""".format(project=_project, data_path=os.path.join(_project, 'outputs', '_large_data.json'),
          out_path=os.path.join(_project, 'outputs', '_large_results.json'))

print('Running methods (subprocess)...')
result = subprocess.run([_venv_py, '-c', test_script], capture_output=True, text=True, timeout=600)
print(result.stdout)
if result.stderr:
    # Filter common warnings
    for line in result.stderr.split('\n'):
        if 'Warning' in line or 'warn' in line.lower():
            continue
        if line.strip():
            print('ERR:', line[:200])
