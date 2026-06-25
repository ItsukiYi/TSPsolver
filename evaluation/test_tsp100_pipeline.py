"""Test #2 Pipeline with TSP-100 trained DIFUSCO model."""
import sys, os, time, glob, numpy as np, torch
_project = os.path.dirname(__file__)

# DIFUSCO TSP-100 model
sys.path.insert(0, os.path.join(_project, 'DIFUSCO-main', 'difusco'))
sys.path.insert(1, os.path.join(_project, 'DIFUSCO-main'))
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
    training_split='data/tsp_problems/tsp100_test.txt',
    validation_split='data/tsp_problems/tsp100_test.txt',
    test_split='data/tsp_problems/tsp100_test.txt',
    batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
    lr_scheduler='cosine-decay', num_epochs=5, num_workers=0,
    validation_examples=4, use_activation_checkpoint=False, fp16=False)
device = torch.device('cuda')

ckpts = glob.glob(os.path.join(_project, 'tsp100_categorical/checkpoints/epoch*.ckpt'))
ckpt = sorted(ckpts)[-1]
print('Checkpoint:', ckpt)
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

# Clear DIFUSCO modules, setup DualOpt
for k in list(sys.modules.keys()):
    if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
sys.path = [os.path.join(_project, 'DualOpt-main')] + [p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
sys.path.append(os.path.join(_project, 'src'))

from utils import load_model
from utils.functions import LCP_TSP, load_problem
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)
revisers = []
for sz in [50, 20, 10]:
    path = os.path.join(_project, 'DualOpt-main', 'pretrained', 'local_%d' % sz, 'epoch-100.pt')
    r, _ = load_model(path, is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

np.random.seed(99)
orig_imprs = []; pipe_imprs = []
for idx in range(3):
    pts = np.random.rand(100, 2).astype(np.float64)
    n = len(pts)

    # DIFUSCO with TSP-100 model
    sys.path = [os.path.join(_project, 'DIFUSCO-main', 'difusco'),
                os.path.join(_project, 'DIFUSCO-main')] + [p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
    from utils.diffusion_schedulers import InferenceSchedule
    from utils.tsp_utils import merge_tours

    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1, n, n).to(device); xt = (xt > 0).long()
        ts = InferenceSchedule(inference_schedule='cosine', T=model.diffusion.T, inference_T=50)
        for i in range(50):
            t1, t2 = ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt = model.categorical_denoise_step(pts_t, xt, t1, device, None, target_t=t2)
        hm = xt.float().cpu().numpy().squeeze() + 1e-6
    tours, _ = merge_tours(hm[np.newaxis,:,:], pts, None, sparse_graph=False, parallel_sampling=1)
    dif_tour = tours[0]

    # Restore DualOpt paths
    sys.path = [os.path.join(_project, 'DualOpt-main')] + [p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
    sys.path.append(os.path.join(_project, 'src'))
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
    from utils.functions import LCP_TSP, load_problem
    from src.algorithms import christofides_with_2opt
    from src.utils import compute_distance_matrix, tour_cost

    gc = lambda inp, pi: load_problem('tsp').get_costs(inp, pi, return_local=True)

    # C+2opt baseline
    tour_c2, _ = christofides_with_2opt(pts, max_2opt_iterations=100)
    init_cost = tour_cost(compute_distance_matrix(pts), tour_c2)
    it = tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
    seeds = torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)

    # Original DualOpt
    s = seeds.clone()
    for rid in range(3):
        s = LCP_TSP(s, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
    cost_orig = (s[:,1:]-s[:,:-1]).norm(p=2,dim=2).sum(1)+(s[:,0]-s[:,-1]).norm(p=2,dim=1)

    # #2 Pipeline with TSP-100 model
    dif_tour = [int(v) for v in dif_tour]
    if dif_tour[-1] == dif_tour[0]: dif_tour = dif_tour[:-1]
    seeds_d = torch.from_numpy(pts[dif_tour]).float().unsqueeze(0).to(device)
    s_d = seeds_d.clone()
    for rid in range(3):
        s_d = LCP_TSP(s_d, gc, revisers[rid], [50,20,10][rid], [25,10,5][rid])
    cost_pipe = (s_d[:,1:]-s_d[:,:-1]).norm(p=2,dim=2).sum(1)+(s_d[:,0]-s_d[:,-1]).norm(p=2,dim=1)

    io = (init_cost - cost_orig.item()) / init_cost * 100
    ip = (init_cost - cost_pipe.item()) / init_cost * 100
    print('[%d] C+2opt:%.2f  Orig:%.2f(%+.1f%%)  #2(TSP100):%.2f(%+.1f%%)' % (
        idx+1, init_cost, cost_orig.item(), io, cost_pipe.item(), ip))
    orig_imprs.append(io); pipe_imprs.append(ip)

print('\nMean: Orig=%+.2f%%  #2(TSP100)=%+.2f%%' % (np.mean(orig_imprs), np.mean(pipe_imprs)))
