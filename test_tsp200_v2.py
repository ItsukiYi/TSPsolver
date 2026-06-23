"""Test #2 with TSP-200 trained DIFUSCO (sparse mode)."""
import sys, os, glob, numpy as np, torch
_p = os.path.dirname(__file__)
device = torch.device('cuda')

# ---- DIFUSCO ----
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
    training_split='data/tsp_problems/tsp200_test.txt',
    validation_split='data/tsp_problems/tsp200_test.txt',
    test_split='data/tsp_problems/tsp200_test.txt',
    batch_size=1, learning_rate=2e-4, weight_decay=1e-4,
    lr_scheduler='cosine-decay', num_epochs=5, num_workers=0,
    validation_examples=2, use_activation_checkpoint=False, fp16=False)
ckpt = sorted(glob.glob(os.path.join(_p,'tsp200_categorical/checkpoints/epoch*.ckpt')))[-1]
print('Checkpoint:', ckpt)
model = TSPModel.load_from_checkpoint(ckpt, param_args=args, strict=False)
model = model.to(device); model.eval()

# ---- DualOpt ----
for k in list(sys.modules.keys()):
    if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
sys.path = [os.path.join(_p,'DualOpt-main')]+[p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
sys.path.append(os.path.join(_p,'src'))
from utils import load_model
from utils.functions import LCP_TSP, load_problem
from src.algorithms import christofides_with_2opt
from src.utils import compute_distance_matrix, tour_cost

gc = lambda inp,pi: load_problem('tsp').get_costs(inp,pi,return_local=True)
revisers = []
for sz in [50,20,10]:
    r,_ = load_model(os.path.join(_p,'DualOpt-main','pretrained','local_'+str(sz),'epoch-100.pt'), is_local=True)
    r.to(device); r.eval(); r.set_decode_type('greedy')
    revisers.append(r)

np.random.seed(99)
for idx in range(3):
    pts = np.random.rand(200,2).astype(np.float64); n = 200

    # Generate DIFUSCO heatmap + merge
    sys.path = [os.path.join(_p,'DIFUSCO-main','difusco'),os.path.join(_p,'DIFUSCO-main')]+[p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
    from utils.diffusion_schedulers import InferenceSchedule as IS
    from utils.tsp_utils import merge_tours as mt
    with torch.no_grad():
        pts_t = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        xt = torch.randn(1,n,n).to(device); xt=(xt>0).long()
        ts = IS(inference_schedule='cosine',T=model.diffusion.T,inference_T=50)
        for i in range(50):
            t1,t2=ts(i); t1=np.array([t1]).astype(int); t2=np.array([t2]).astype(int)
            xt=model.categorical_denoise_step(pts_t,xt,t1,device,None,target_t=t2)
        hm=xt.float().cpu().numpy().squeeze()+1e-6
    tours,_=mt(hm[np.newaxis,:,:],pts.astype(np.float64),None,sparse_graph=False,parallel_sampling=1)
    dif_tour=[int(v) for v in tours[0]]

    # Back to DualOpt
    sys.path=[os.path.join(_p,'DualOpt-main')]+[p for p in sys.path if 'difusco' not in p.lower() and 'DualOpt' not in p]
    sys.path.append(os.path.join(_p,'src'))
    for k in list(sys.modules.keys()):
        if k.startswith('utils') or k.startswith('pl_'): del sys.modules[k]
    from utils.functions import LCP_TSP as lcp
    from src.algorithms import christofides_with_2opt as c2
    from src.utils import compute_distance_matrix as cdm, tour_cost as tc

    tour_c2,_=c2(pts,max_2opt_iterations=50)
    init_cost=tc(cdm(pts),tour_c2)
    it=tour_c2[:-1] if tour_c2[-1]==tour_c2[0] else tour_c2
    seeds=torch.from_numpy(pts[it]).float().unsqueeze(0).to(device)

    s=seeds.clone()
    for rid in range(3):
        s=lcp(s,gc,revisers[rid],[50,20,10][rid],[25,10,5][rid])
    cost_orig=(s[:,1:]-s[:,:-1]).norm(p=2,dim=2).sum(1)+(s[:,0]-s[:,-1]).norm(p=2,dim=1)

    if dif_tour[-1]==dif_tour[0]: dif_tour=dif_tour[:-1]
    seeds_d=torch.from_numpy(pts[dif_tour]).float().unsqueeze(0).to(device)
    s_d=seeds_d.clone()
    for rid in range(3):
        s_d=lcp(s_d,gc,revisers[rid],[50,20,10][rid],[25,10,5][rid])
    cost_pipe=(s_d[:,1:]-s_d[:,:-1]).norm(p=2,dim=2).sum(1)+(s_d[:,0]-s_d[:,-1]).norm(p=2,dim=1)

    io=(init_cost-cost_orig.item())/init_cost*100
    ip=(init_cost-cost_pipe.item())/init_cost*100
    print('[%d] C+2opt:%.2f  Orig:%+.1f%%  #2(TSP200):%+.1f%%'%(idx+1,init_cost,io,ip))
