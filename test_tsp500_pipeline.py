"""Test Improvement #2 on TSP-500."""
import sys, os, time, pickle, numpy as np, torch, subprocess

_project = os.path.dirname(__file__)
_venv_py = os.path.join(_project, 'venv', 'Scripts', 'python.exe')
_out = os.path.join(_project, 'outputs', '_tsp500.pkl')

# Step 1: DIFUSCO heatmaps
gen_code = '''import sys, os, pickle, numpy as np, torch
_p = r"''' + _project + r'''"
sys.path.insert(0, os.path.join(_p, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_p, 'DIFUSCO-main'))
from pl_tsp_model import TSPModel
from argparse import Namespace
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import merge_tours

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

np.random.seed(42)
pts_list = [np.random.rand(500, 2).astype(np.float64) for _ in range(3)]
data = {'pts': [], 'hm': [], 'tours': []}
for pts in pts_list:
    n = len(pts)
    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        hm = xt.float().cpu().numpy().squeeze() + 1e-6
    tours, _ = merge_tours(hm[np.newaxis,:,:], pts, None, sparse_graph=False, parallel_sampling=1)
    data['pts'].append(pts); data['hm'].append(hm); data['tours'].append(tours[0])
    print('Instance done')

with open(r"''' + _out + r'''", 'wb') as f:
    pickle.dump(data, f)
print('All saved')
'''

print('Step 1: DIFUSCO heatmaps for TSP-500 (3 instances)...')
r = subprocess.run([_venv_py, '-c', gen_code], capture_output=True, text=True, timeout=600)
print(r.stdout)
if r.stderr:
    for line in r.stderr.split('\n'):
        if 'Warning' not in line and 'warn' not in line.lower() and 'DIFUSCO' not in line and line.strip():
            if 'triton' not in line and 'torch_geometric' not in line and 'torch_scatter' not in line:
                print('ERR:', line[:150])
if r.returncode != 0:
    print('FAILED'); sys.exit(1)

# Step 2: Run methods
print('\nStep 2: Running DualOpt methods...')
sys.path.insert(0, os.path.join(_project, 'DualOpt-main'))
sys.path.append(os.path.join(_project, 'src'))
from utils import load_model
from utils.functions import LCP_TSP, load_problem
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

with open(_out, 'rb') as f:
    data = pickle.load(f)

device = torch.device('cuda')
gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
revisers = []
for sz in [50, 20, 10]:
    path = os.path.join(_project, 'DualOpt-main', 'pretrained', 'local_%d' % sz, 'epoch-100.pt')
    r, _ = load_model(path, is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

print('\nTSP-500 results:')
orig_imprs = []; pipe_imprs = []
for idx in range(3):
    pts = data['pts'][idx]
    dif_tour = data['tours'][idx]
    t0 = time.time()

    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=50)
    init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
    it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)

    s = seeds.clone()
    for rid in range(3):
        s = LCP_TSP(s, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
    cost_orig = (s[:,1:]-s[:,:-1]).norm(p=2,dim=2).sum(1)+(s[:,0]-s[:,-1]).norm(p=2,dim=1)

    if dif_tour[-1] == dif_tour[0]: dif_tour = dif_tour[:-1]
    seeds_d = torch.from_numpy(pts[dif_tour]).float().unsqueeze(0).to(device)
    s_d = seeds_d.clone()
    for rid in range(3):
        s_d = LCP_TSP(s_d, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
    cost_pipe = (s_d[:,1:]-s_d[:,:-1]).norm(p=2,dim=2).sum(1)+(s_d[:,0]-s_d[:,-1]).norm(p=2,dim=1)

    io = (init_cost - cost_orig.item()) / init_cost * 100
    ip = (init_cost - cost_pipe.item()) / init_cost * 100
    t1 = time.time()
    print('  [%d] C+2opt:%.2f  Orig:%.2f(%+.1f%%)  #2-Pipe:%.2f(%+.1f%%)  [%.0fs]' % (
        idx+1, init_cost, cost_orig.item(), io, cost_pipe.item(), ip, t1-t0))
    orig_imprs.append(io); pipe_imprs.append(ip)

print('\n  Mean: Orig=%+.2f%%  #2-Pipe=%+.2f%%' % (np.mean(orig_imprs), np.mean(pipe_imprs)))
