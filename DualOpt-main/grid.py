# import matplotlib.pyplot as plt
import lkh
import math
import numpy as np
import tsplib95
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from multiprocessing import Pool
import pickle
import warnings
import torch
import os
warnings.filterwarnings("ignore")

time_parts = np.zeros(5)

def sample2tsplib(sample : np.ndarray, tsplib = False):
    """
        sample(np.ndarray) : all nodes' coordinates in a instance, the shape is(n_nodes, 2)
        return :
            problem_str(str) : 
    """
    dimension = sample.shape[0] #nodes number
    problem_str = f"""NAME : tsp
COMMENT : tsp_comm
TYPE : TSP
DIMENSION : {dimension}
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION"""
    if tsplib == False:
        for idx in range(dimension):
            problem_str += "\n{} {} {}".format(idx + 1, sample[idx][0]*1000 , sample[idx][1]*1000 )
    else:
        for idx in range(dimension):
            problem_str += "\n{} {} {}".format(idx + 1, sample[idx][0], sample[idx][1])
    problem_str += "\nEOF\n"
    return problem_str

def sampleWithLinks2tsplib(sample, in_region_idxs, links, tsplib = False):
    """
        sample(np.ndarray) : all nodes' coordinates in a instance, the shape is(n_nodes, 2)
        return :
            problem_str(str) : 
    """
    dimension = len(in_region_idxs) #nodes number
    idx_map = {}
    for idx, i in enumerate(in_region_idxs):
        idx_map[i] = idx
    
    problem_str = f"""NAME : tsp
COMMENT : tsp_comm
TYPE : TSP
DIMENSION : {dimension}
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION"""
    if tsplib == False:
        for idx in range(len(in_region_idxs)):
            problem_str += "\n{} {} {}".format(idx + 1, sample[in_region_idxs[idx]][0]*1000 , sample[in_region_idxs[idx]][1]*1000 )
    else:
        for idx in range(dimension):
            problem_str += "\n{} {} {}".format(idx + 1, sample[in_region_idxs[idx]][0] , sample[in_region_idxs[idx]][1] )
#     problem_str = f"""NAME : tsp
# COMMENT : tsp_comm
# TYPE : TSP
# DIMENSION : {dimension}
# EDGE_WEIGHT_TYPE: EXPLICIT
# EDGE_WEIGHT_FORMAT: FULL_MATRIX 
# EDGE_WEIGHT_SECTION"""

    # dist = []
    # for i in range(len(in_region_idxs)):
    #     dist_list = []
    #     for j in range(len(in_region_idxs)):
    #         if i == j:
    #             dist_list.append('9999')
    #         else:
    #             dist_list.append(str(node_distance(sample[in_region_idxs[i]], sample[in_region_idxs[j]])*1000))
    #     dist.append(dist_list)
    
    # for l in links:
    #     for i in range(len(in_region_idxs)):
    #         if i != l[1]:
    #             dist[idx_map[l[0]]][i] = "9999"

    #     dist[idx_map[l[0]]][idx_map[l[1]]] = "0"
    #     dist[idx_map[l[1]]][idx_map[l[0]]] = "0"

    # for d in dist:
    #     dist_str = "    ".join(d)
    #     problem_str += f'\n {dist_str}'

    if len(links) > 0:
        problem_str += "\nFIXED_EDGES_SECTION"
        for l in links:
            if l[0] in idx_map and l[1] in idx_map:
                problem_str += f'\n{str(idx_map[l[0]] + 1)} {str(idx_map[l[1]] + 1)}'
        problem_str += f'\n-1'
    problem_str += "\nEOF\n"

    return problem_str

def node_distance(left, right):
    return math.sqrt((left[0]-right[0])**2 + (left[1]-right[1])**2)

def cal_route_distance(pos, route):
    dist = 0
    for i in range(len(route) - 1):
        dist += node_distance(pos[route[i]], pos[route[i + 1]])
    return dist

def cal_route_distance_matrix(matrix, route):
    dist = 0
    for i in range(len(route) - 1):
        dist += matrix[route[i]][route[i + 1]]
    return dist

def lkh_solver(sample, runs = 1):
    problem = tsplib95.parse(sample2tsplib(sample))

    # routes = lkh.solve('/root/autodl-tmp/zsp/lcp/VSR-LKH', problem = problem, max_trials = sample.shape[0], runs = runs)
    routes = lkh.solve('LKH.exe', problem = problem, max_trials = sample.shape[0], runs = 10)
    route = [r - 1 for r in routes[0]] # to index-0
    route.append(0)

    #calculate distance
    dist = cal_route_distance(sample, route)
    return route, dist

def plot_tour(pos, tour, len_idx):
    x = pos[:, 0]
    y = pos[:, 1]
    plt.scatter(x, y, s = 5)
    for i in range(len(tour) - 1):
        plt.arrow(x[tour[i]],
                  y[tour[i]],
                  x[tour[i + 1]] - x[tour[i]],
                  y[tour[i + 1]] - y[tour[i]],
                  length_includes_head=True,
                  head_width=0.005,
                  linewidth=0.5)
    plt.savefig('visualization'+ str(len_idx) +'.jpg', dpi=1000)



def extract_sample(sample, region, mask):
    in_region_idxs, out_region_idxs = [],[] # index of exterior region and interior region in sample
    lbx, lby = region[0] # coordinate of left-bottom
    rtx, rty = region[1] # coordinate of right-up
    for idx in range(sample.shape[0]):
        if idx in mask:
            continue
        x, y = sample[idx]
        if lbx <= x <= rtx and lby <= y <= rty:
            in_region_idxs.append(idx)
        else:
            out_region_idxs.append(idx)
    return in_region_idxs, out_region_idxs


def split_tour(sample, tour, region):
    subtour, cur_tour = [], []
    lbx, lby = region[0]
    rtx, rty = region[1]
    for idx in tour:
        x,y = sample[idx]
        if lbx <= x <= rtx and lby <= y <= rty:
            cur_tour.append(idx)
        elif 0 < len(cur_tour) <=2:
            cur_tour = []
        elif len(cur_tour) > 2:
            subtour.append(cur_tour)
            cur_tour = []
    
    return subtour


def remove_mid_nodes(route_list, mask, links):
    """Only the two endpoints of the route within the interior region are retained. 
    These two endpoints are added to the links while the intermediate nodes are masked.

    Args:
        - route_list(list):
        - mask(set):
        - links(list):
    
    Returns:
        - mask(set):
        - new_links(list):
    """
    save_links = set(range(len(links))) # indices of kept link
    ept2idx = {} # The mapping from the endpoints of the link to the index
    for idx, l in enumerate(links): # create ept2idx
        ept2idx[l[0]] = idx
        ept2idx[l[1]] = idx

    new_links = [] 
    for sub_route in route_list: 
        for idx in range(len(sub_route)):
            if idx != 0 and idx != len(sub_route) - 1: # the intermediate nodes are masked.
                mask.add(sub_route[idx])
            if sub_route[idx] in ept2idx and ept2idx[sub_route[idx]] in save_links: #delete duplicate links
                save_links.remove(ept2idx[sub_route[idx]])
        new_links.append([sub_route[0], sub_route[-1]])
    
    save_links = [links[i] for i in save_links]
    new_links.extend(save_links)

    return mask, new_links



def lkh_solver_by_region(sample, links, mask, region_exterior, region_interior, runs = 1):
    """
    args:
        - region_exterior list[list] : [[bottom_left_x, bottom_left_y], [top_right_x, top_right_y]]
    """

    # extract subsample in region_exterior according to mask
    time_0 = time.time()
    in_region_idxs, out_region_idxs = extract_sample(sample, region_exterior, mask)
    
    tmp_in, tmp_out = extract_sample(sample, region_interior, mask)
    
    # solve sample in region_exterior, [n1, n2] in links denotes that (n1, n2) must in route  
    time_1 = time.time()
    # problem = tsplib95.parse(sample2tsplib(sample[in_region_idxs]))
    problem = tsplib95.parse(sampleWithLinks2tsplib(sample, in_region_idxs, links))
    
    # print("samples:", len(in_region_idxs))
    
    time_2 = time.time()
    if len(in_region_idxs) < 3:
        return [], [], set(), []
    
    routes = lkh.solve('LKH.exe', problem = problem, max_trials = len(in_region_idxs), runs = 10)
    route = [in_region_idxs[r - 1] for r in routes[0]] # to index-0
    route.append(in_region_idxs[0]) # make recycle
    
    time_3 = time.time()
    # split route in region_interior
    route_list = split_tour(sample, route, region_interior)
    
    time_4 = time.time()
    # remove mid nodes in subroute
    mask, links = remove_mid_nodes(route_list, mask, links)
    time_5 = time.time()

    # print(time_1 - time_0, time_2 - time_1, time_3 - time_2, time_4 - time_3, time_5 - time_4)
    global time_parts
    time_parts += np.array([time_1 - time_0, time_2 - time_1, time_3 - time_2, time_4 - time_3, time_5 - time_4])

    return route, links, mask, route_list


def our_solver_parallel(sample, layer_number, return_solution = False, runs = 10):
    # dist_matrix = squareform(pdist(sample, metric='euclidean'))
    # route_matrix = np.zeros((sample.shape[0], sample.shape[0]), dtype = "bool")
    route_matrix = [[] for _ in range(sample.shape[0])]
    x_min = sample[:, 0].min()
    x_max = sample[:, 0].max() 
    y_min = sample[:, 1].min()
    y_max = sample[:, 1].max()

    # create region list
    gap = 1 / (2**(layer_number + 2))
    all_region_interior, all_region_exterior = [], []
    for layer_idx in reversed(range(layer_number)):
        region_interior = []
        region_exterior = []
        regions_number = 2**layer_idx
        interval = 1 / (2**layer_idx)
        for i in range(regions_number):
            for j in range(regions_number):
                region_interior.append([[i * interval * (x_max - x_min) + gap * (x_max - x_min) + x_min, 
                                        j * interval * (y_max -y_min) + gap * (y_max - y_min) + y_min], 
                                        [(i + 1) * interval * (x_max - x_min) - gap * (x_max - x_min) + x_min, 
                                        (j + 1) * interval * (y_max -y_min) - gap * (y_max -y_min) + y_min]])
                region_exterior.append([[i * interval * (x_max - x_min) + x_min, 
                                        j * interval * (y_max -y_min) + y_min], 
                                        [(i + 1) * interval * (x_max - x_min) + x_min, 
                                        (j + 1) * interval * (y_max -y_min) + y_min]])
                # print(region_exterior)
        all_region_interior.append(region_interior)
        all_region_exterior.append(region_exterior)

    route_all, links = [],[]
    mask = set()

    for layer_idx in range(layer_number):
        t= time.time()
        pool = Pool(processes = 8)
        results = []
        for idx in range(len(all_region_interior[layer_idx])):
            region_exterior = all_region_exterior[layer_idx][idx]
            region_interior = all_region_interior[layer_idx][idx]
            # print(idx, region_exterior, region_interior)
            results.append(pool.apply_async(lkh_solver_by_region, (sample, links, mask, region_exterior, region_interior, runs)))
        pool.close()
        pool.join()
        
        for res in results:
            route, links_cur, mask_cur, route_list = res.get()
            links.extend(links_cur)
            mask.update(mask_cur)

            if layer_idx != layer_number - 1: # not need last route list
                route_all.extend(route_list)

            # for r in route_list:
            #     dist_matrix[r[0]][r[-1]] = cal_route_distance_matrix(dist_matrix, r)
            #     dist_matrix[r[-1]][r[0]] = cal_route_distance_matrix(dist_matrix, r)

    # ours_cost = cal_route_distance_matrix(dist_matrix, route)

    if return_solution:
        for i in range(len(route) - 1):
            # route_matrix[route[i]][route[i + 1]] = True
            # route_matrix[route[i + 1]][route[i]] = True
            route_matrix[route[i]].append(route[i + 1])
            route_matrix[route[i + 1]].append(route[i])

        for r in route_all[::-1]:
            # route_matrix[r[0]][r[-1]] = False
            # route_matrix[r[-1]][r[0]] = False
            route_matrix[r[0]].remove(r[-1])
            route_matrix[r[-1]].remove(r[0])
            for i in range(len(r) - 1):
                # route_matrix[r[i]][r[i + 1]] = True
                # route_matrix[r[i + 1]][r[i]] = True
                route_matrix[r[i]].append(r[i + 1])
                route_matrix[r[i + 1]].append(r[i])

        final_route = [0]
        # net = np.where(route_matrix[0] == True)[0][0]
        net = route_matrix[0][0]
        final_route.append(net)
        for i in range(sample.shape[0]-2):
            # nxt = np.where(route_matrix[final_route[-1]] == True)[0]
            nxt = route_matrix[final_route[-1]]
            for n in nxt:
                if n!=final_route[-2]:
                    final_route.append(n)
                    break
        final_route.append(0)

        # check solution
        for i in range(sample.shape[0]):
            if not i in final_route:
                print("invaild solution")
                exit()
        assert len(final_route) == sample.shape[0] + 1, "invaild solution"
        ours_cost = cal_route_distance(sample, final_route)

    if return_solution:
        return final_route, ours_cost
    else:
        return ours_cost



def our_solver(sample, return_solution = False):
    # dist_matrix = squareform(pdist(sample, metric='euclidean'))
    route_matrix = np.zeros((sample.shape[0], sample.shape[0]), dtype = "bool")

    # create region list
    interval_interior, interval_exterior = 0.1, 0.1
    all_region_interior, all_region_exterior = [], []
    num_iter = math.ceil((0.5 - interval_exterior) / interval_interior)
    cur_interior = 0.5
    for i in range(num_iter):
        cur_interior = max(cur_interior - interval_interior, 0)
        cur_exterior = cur_interior - interval_exterior
        all_region_interior.append([[cur_interior, cur_interior], [1 - cur_interior, 1 - cur_interior]])
        all_region_exterior.append([[cur_exterior, cur_exterior], [1 - cur_exterior, 1 - cur_exterior]])

    route_all, links = [],[]
    mask = set()

    for idx in range(len(all_region_interior)):
        region_exterior = all_region_exterior[idx]
        region_interior = all_region_interior[idx]
        route, links, mask, route_list = lkh_solver_by_region(sample, links, mask, region_exterior, region_interior, runs = 10)

        if idx != len(all_region_interior) - 1: # not need last route list
            route_all.extend(route_list)
        # for r in route_list:
        #     dist_matrix[r[0]][r[-1]] = cal_route_distance_matrix(dist_matrix, r)
        #     dist_matrix[r[-1]][r[0]] = cal_route_distance_matrix(dist_matrix, r)

    # ours_cost = cal_route_distance_matrix(dist_matrix, route)

    if return_solution:
        for i in range(len(route) - 1):
            route_matrix[route[i]][route[i + 1]] = True
            route_matrix[route[i + 1]][route[i]] = True

        for r in route_all[::-1]:
            route_matrix[r[0]][r[-1]] = False
            route_matrix[r[-1]][r[0]] = False
            for i in range(len(r) - 1):
                route_matrix[r[i]][r[i + 1]] = True
                route_matrix[r[i + 1]][r[i]] = True

        final_route = [0]
        net = np.where(route_matrix[0] == True)[0][0]
        final_route.append(net)
        for i in range(sample.shape[0]-2):
            nxt = np.where(route_matrix[final_route[-1]] == True)[0]
            for n in nxt:
                if n!=final_route[-2]:
                    final_route.append(n)
                    break
        final_route.append(0)

        # check solution
        # 1. Each vertex must be visited
        for i in range(sample.shape[0]):
            if not i in final_route:
                print("invaild solution")
                exit()
        # 2. Visit each vertex once
        if (sample.shape[0] != len(final_route) - 1):
            print("invaild solution")
            exit()
            
        ours_cost = cal_route_distance(sample, final_route)

    if return_solution:
        return final_route, ours_cost
    else:
        return ours_cost

def load_data(filepath):
	assert os.path.exists(filepath), print('Error: filepath {} not exist!'.format(filepath))
	load_file = open(filepath, 'rb')
	load_data = pickle.load(load_file)
	load_file.close()
	return load_data

def load_instance(filename, scale=True):
	problem = tsplib95.load(filename)
	node_indices = list(problem.get_nodes())
	data = [problem.node_coords[node_idx] for node_idx in node_indices]
	data = torch.tensor(data).double()
    
	# if scale:
	# 	d_min = data.min(dim=0)[0]
	# 	factor = max(data.max(dim=0)[0] - d_min)
	# 	data = data - d_min
	return data	

def readDataFile(filePath):
    """
    read datafile 
    """
    res = []
    sols = []
    with open(filePath, "r") as fp:
        datas = fp.readlines()
        for data in datas:
            tmp = data.split("output")
            data = [float(i) for i in tmp[0].split()]
            loc_x = np.array(data[::2])
            loc_y = np.array(data[1::2])
            data = np.stack([loc_x, loc_y], axis = 1)
            res.append(data)
            if len(tmp) > 1:
                sol = [int(i) - 1 for i in tmp[1].split()]
                sols.append(sol)
    res = np.stack(res, axis = 0)
    return res, sols


def readTSPLib(filePath):
    """
        read TSPLib
    """
    data_trans, data_raw = [], []
    with open(filePath, "r") as fp:
        loc_x = []
        loc_y = []
        datas = fp.readlines()
        for data in datas:
            if ":" in data or "EOF" in data or "NODE_COORD_SECTION" in data:
                continue
            data = [float(i) for i in data.split()]
            if len(data) == 3:
                loc_x.append(data[1])
                loc_y.append(data[2])
        loc_x = np.array(loc_x)
        loc_y = np.array(loc_y)

        data = np.stack([loc_x, loc_y], axis=1)
        data_raw.append(data)

        mx = loc_x.max() - loc_x.min()
        my = loc_y.max() - loc_y.min()
        data = np.stack([loc_x - loc_x.min(), loc_y - loc_y.min()], axis = 1)
        data = data / max(mx, my)
        data_trans.append(data)

    data_trans = np.stack(data_trans, axis = 0)
    data_raw = np.stack(data_raw, axis = 0)
    return data_trans, data_raw

def initialSolFile(ours_solution):
    with open('initialSol.txt', 'w') as f:
        f.write('TOUR_SECTION :' + '\n')
        for i in ours_solution:
            f.write(str(i) + '\n')  
        f.write('-1')  # 写入结束符号


def segment_revise(sample, ours_solution, seg_len, runs = 10):
    seg_num = sample.shape[0] // seg_len
    if (sample.shape[0] % seg_len != 0):
        seg_num += 1
    routes = []
    
    for i in range(seg_num):
        links = []
        if (i == seg_num - 1):
            seg_initial_sol = ours_solution[i * seg_len: ]
        else:
            seg_initial_sol = ours_solution[i * seg_len: (i + 1) * seg_len]
        links.append([seg_initial_sol[0], seg_initial_sol[-1]])
        problem = tsplib95.parse(sampleWithLinks2tsplib(sample, seg_initial_sol, links))
        initialSolFile(range(0, len(seg_initial_sol)))
        seg_routes = lkh.solve('LKH.exe', problem = problem, max_trials = len(seg_initial_sol), initial_tour_file = 'initialSol.txt', runs = 200)
        seg_routes = [seg_initial_sol[r - 1] for r in seg_routes[0]] # to index-0
        routes += seg_routes
        assert seg_routes[0] == seg_initial_sol[0] and seg_routes[-1] == seg_initial_sol[-1], "invaild segment"
        
    routes.append(routes[0]) # make a cycle
    
    # check solution
    for i in range(sample.shape[0]):
        if not i in routes:
            print("invaild solution")
            exit()
    assert len(routes) == sample.shape[0] + 1, "invaild solution"
    
    ours_cost = cal_route_distance(sample, routes)
    
    return routes, ours_cost

def first_step(samples, lkh_layer_number, solutions, val_size):
    samples_raw = samples
    all_concorde_cost = []
    all_lkh_cost, all_lkh_time = [],[]
    all_ours_cost, all_ours_time = [],[]
    samples_solution = []
    print("*"*30)
    for idx in range(val_size):
        # print("solve sample :{}".format(idx + 1))

        sample = samples[idx]
        sample_raw = samples_raw[idx]
        
        # ours
        time_start = time.time()
        ours_solution, ours_cost = our_solver_parallel(sample, lkh_layer_number,return_solution = True, runs = 10)
        samples_solution.append(np.array(ours_solution[:-1]))
            
        ours_cost = cal_route_distance(sample_raw, ours_solution)
        time_end = time.time()
        ours_time = time_end - time_start
        all_ours_cost.append(ours_cost)
        all_ours_time.append(ours_time)
        print("solve sample :{}, {:<15}:{:.5f}, total time:{:.5f}".format(idx + 1, "ours cost", ours_cost, ours_time))
        
    return samples_solution

    # if (not solutions is None) and len(solutions) > 0:
    #     print("Concorde average cost:{:.3f}".format(sum(all_concorde_cost)/len(all_concorde_cost)))
    # print("LKH3 average cost:{:.3f}, total time:{:3f}".format(sum(all_lkh_cost)/len(all_lkh_cost), sum(all_lkh_time)))
    print("Ours average cost:{:.3f}, total time:{:3f}".format(sum(all_ours_cost)/len(all_ours_cost), sum(all_ours_time)))
    
