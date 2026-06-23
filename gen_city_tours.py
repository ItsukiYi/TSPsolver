"""Compute all tours for city scenario and save to pickle."""

import sys, os, pickle, json, time, numpy as np, torch
_p = os.path.dirname(__file__)
os.environ['PATH'] = os.path.join(_p, 'DualOpt-main', 'LKH-3.0.7') + os.pathsep + os.environ['PATH']
sys.path.append(os.path.join(_p, 'src'))

from src.algorithms import nearest_neighbor_tsp, christofides_tsp, christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost
from city_delivery_scenario import generate_city_scenario

pts = generate_city_scenario(500, seed=42)
n = len(pts)
dist_mat = compute_distance_matrix(pts)
tours = {}

# NN
t, _ = nearest_neighbor_tsp(pts); tours['NN'] = t

# Christofides
t, _ = christofides_tsp(pts); tours['Christofides'] = t

# C+2opt
t, _ = christofides_with_2opt(pts, max_2opt_iterations=500); tours['C+2opt'] = t

# DualOpt
sys.path.insert(0, os.path.join(_p, 'DualOpt-main'))
from utils import load_model; from utils.functions import LCP_TSP, load_problem
device = torch.device('cuda')
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
revisers = []
for sz in [50,20,10]:
    r,_ = load_model(os.path.join(_p,'DualOpt-main','pretrained','local_%d'%sz,'epoch-100.pt'), is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)
it = tours['C+2opt'][:-1]
seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)
for rid in range(3):
    rlen = [50,20,10][rid]
    if rlen < seeds.shape[1]: seeds = LCP_TSP(seeds, gc, revisers[rid], rlen, [25,10,5][rid])
from scipy.spatial import KDTree
tree = KDTree(pts); _, idx = tree.query(seeds[0].cpu().numpy())
dt = idx.tolist()
if dt[0] != dt[-1]: dt.append(dt[0])
tours['DualOpt'] = dt

# LKH3
import lkh, tsplib95
problem = tsplib95.models.StandardProblem()
problem.name='TSP'; problem.type='TSP'; problem.dimension=n
problem.edge_weight_type='EUC_2D'
problem.node_coords={i+1:(float(pts[i][0]*1e6),float(pts[i][1]*1e6)) for i in range(n)}
sol = lkh.solve('LKH.exe', problem=problem, max_trials=100, runs=5)
t = [r-1 for r in sol[0]]
if t[0]!=t[-1]: t.append(t[0])
tours['LKH3'] = t

# DIFUSCO via subprocess
import subprocess, glob
_venv_py = os.path.join(_p, 'venv', 'Scripts', 'python.exe')
ckpt = sorted(glob.glob(os.path.join(_p, 'tsp50_categorical/checkpoints/epoch=6-step*.ckpt')))[0]
r = subprocess.run([_venv_py, '-c', '''
import sys, os, json, numpy as np, torch
_p = r"''' + _p + r'''"
sys.path.insert(0, os.path.join(_p,'DIFUSCO-main','difusco'))
sys.path.insert(1, os.path.join(_p,'DIFUSCO-main'))
from pl_tsp_model import TSPModel
from argparse import Namespace
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import merge_tours, batched_two_opt_torch
args = Namespace(diffusion_type='categorical', diffusion_schedule='cosine',
    diffusion_steps=1000, inference_diffusion_steps=50,
    inference_schedule='cosine', inference_trick='ddim',
    n_layers=12, hidden_dim=256, sparse_factor=-1, aggregation='sum',
    two_opt_iterations=1000, parallel_sampling=1, sequential_sampling=1,
    save_numpy_heatmap=False, storage_path='.',
    training_split='data/tsp_problems/tsp50_test.txt',
    validation_split='data/tsp_problems/tsp50_test.txt',
    test_split='data/tsp_problems/tsp50_test.txt',
    batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
    lr_scheduler='cosine-decay', num_epochs=50, num_workers=0,
    validation_examples=8, use_activation_checkpoint=False, fp16=False)
device = torch.device('cuda')
model = TSPModel.load_from_checkpoint(r"''' + ckpt.replace('\\','/') + r'''", param_args=args, strict=False)
model = model.to(device); model.eval()
pts = np.array(''' + json.dumps(pts.tolist()) + r''', dtype=np.float64); n = len(pts)
with torch.no_grad():
    pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    xt = torch.randn(1,n,n).to(device); xt=(xt>0).long()
    ts = InferenceSchedule(inference_schedule='cosine',T=model.diffusion.T,inference_T=50)
    for i in range(50):
        t1,t2=ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
        xt=model.categorical_denoise_step(pts_t,xt,t1,device,None,target_t=t2)
    hm=xt.float().cpu().numpy().squeeze()+1e-6
np_pts=pts.astype(np.float64)
tours_d,_=merge_tours(hm[np.newaxis,:,:],np_pts,None,sparse_graph=False,parallel_sampling=1)
solved,_=batched_two_opt_torch(np_pts,np.array(tours_d).astype('int64'),max_iterations=1000,device=device)
print(json.dumps(solved[0].tolist()))
'''], capture_output=True, text=True, timeout=300)
tours['DIFUSCO'] = json.loads(r.stdout.strip().split('\n')[-1])

# Save
with open(os.path.join(_p, 'outputs', '_city_tours.pkl'), 'wb') as f:
    pickle.dump(tours, f)

print('Costs:')
for k in ['NN','Christofides','C+2opt','DIFUSCO','DualOpt','LKH3']:
    print('  %s: %.2f' % (k, tour_cost(dist_mat, tours[k])))
print('Saved tours to _city_tours.pkl')
