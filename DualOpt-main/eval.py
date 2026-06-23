import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature, second_step
from nets.attention_model import set_decode_type
from grid import first_step, load_data
import pickle
from utils.functions import load_problem
import pprint as pp


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    revisers = []
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f'pretrained/local_{reviser_size}/epoch-100.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type("greedy")
     
    dataset = reviser.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
    
    results, t1, t2 = _eval_dataset(dataset, width, softmax_temp, opts, device,revisers)

    parallelism = opts.eval_batch_size

    costs_original, costs_revised = zip(*results) 
    costs_original = torch.cat(costs_original, dim=0)
    costs_revised = torch.cat(costs_revised, dim=0)
    print("Average costs_first_step: {} +- {}".format(costs_original.mean().item(), 
                            (2 * torch.std(costs_original) / math.sqrt(len(costs_original))).item()))
    print("Average cost_second_step: {} +- {}".format(costs_revised.mean().item(), 
                            (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))

    print("Calculated total duration: {} + {}".format(t1, t2))
    return 

def _eval_dataset(dataset, width, softmax_temp, opts, device,revisers):
    time1 = time.time()

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
   
    pi_all = first_step(dataset.samples, opts.lkh_layer_number, dataset.solutions, opts.val_size) 
    pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(1, opts.val_size, opts.problem_size) 
    
    time2 = time.time()
    
    results = []
    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)

                p_size = batch.size(1)
                batch = batch.repeat(1, 1, 1) # (1,1,1) for pctsp
                pi_batch = pi_all[:, batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size, :].reshape(-1, p_size)# pi_all： width=1, val_size, p_size
                seeds = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1,1,2))
                seeds = seeds.to(device).float() # (bs, problem_size, 2)
                
                get_cost_func = lambda input, pi: load_problem(opts.problem).get_costs(input, pi, return_local=True)
                
                costs_original, costs_revised = second_step(seeds, get_cost_func, opts, revisers=revisers)
                

        time3 = time.time()

        results.append((costs_original,costs_revised))
    return results, time2 - time1, time3 - time2


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=2000)
    parser.add_argument("--dataset_path", type=str, help="Filename of the dataset(s) to evaluate")
    parser.add_argument("--res_path", type=str)
    parser.add_argument("--lkh_layer_number", type=int, default=2)
    parser.add_argument("--decode_strategy", type=str,default="greedy")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=16,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=2,
                        help="Softmax temperature (sampling or bs)")
    
    parser.add_argument('--revision_lens', nargs='+', default=[50, 20, 10] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[25, 10, 5], type=int,
                        help='Revision iterations (I_n)')
    
    parser.add_argument('--problem', default='tsp', type=str)
    parser.add_argument('--width', type=int, default=0,help='number of candidate solutions (M)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

  
    directory = "Dataset/random"
    
    opts = parser.parse_args()
    opts.dataset_path = f'Dataset/random/tsp{opts.problem_size}_test_concorde.txt'
    opts.res_path = f'Dataset/random/first_step_result/tsp{opts.problem_size}_solution_first_step.pkl'
            
    print()
    print('*'*80)
    print()
    # print(opts.dataset_path)
    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
    "Cannot specify result filename with more than one dataset or more than one width"
    eval_dataset(opts.dataset_path, opts.width, opts.softmax_temperature, opts)
     
    