import dgl
import re
import numpy as np
import pickle
import math
import itertools
import networkx as nx
from utils import groupby, car, cadr, cdr, info, load

def gen_data(gdef, prof_data, batchsize, devices, intra=2810, inter=2810):
    edge_link = [], []
    link_feats = []
    device_feats = [[memory / 10_000_000_000] for _, _, memory in devices]
    tasks = {}
    for i, (name, *_) in enumerate(devices):
        task = re.search("task:(\d+)/", name)[1]
        if task in tasks:
            for other in tasks[task]:
                edge_link[0].append(i)
                edge_link[1].append(other)
                edge_link[0].append(other)
                edge_link[1].append(i)
                link_feats.append([0, intra / 100_000, math.log(intra) / 10])
                link_feats.append([0, intra / 100_000, math.log(intra) / 10])
            tasks[task].append(i)
        else:
            tasks[task] = [i]
    for task, devs in tasks.items():
        for dev in devs:
            for another_task, other_devs in tasks.items():
                if another_task != task:
                    for another_dev in other_devs:
                        edge_link[0].append(dev)
                        edge_link[1].append(another_dev)
                        edge_link[0].append(another_dev)
                        edge_link[1].append(dev)
                        link_feats.append([1, inter / 100_000, math.log(inter) / 10])
                        link_feats.append([1, inter / 100_000, math.log(inter) / 10])

    base_nccl_model = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152]
    nccl_models = {}
    dgroups = groupby(devices, key=lambda x: re.search("task:(\d+)/", x[0])[1], value=lambda x: x[0])

    for task, devs in dgroups.items():
        nccl_models[','.join(sorted(devs))] = [ x * 2810 / intra for x in base_nccl_model ]

    for tasks in (t for i in range(2, len(dgroups)+1) for t in itertools.combinations(dgroups.keys(), i)):
        devs = [dgroups[t][0] for t in tasks] # the first (alphabet order) device is the leader of the task
        nccl_models[','.join(sorted(devs))] = [ x * 2810 / inter for x in base_nccl_model ]

    def group_with_topk_layers(n_groups=20):
        # NOTE: the format of basegroups is like [0, 1,2,3,4,3,2]. i.e. the i-th element is the group id of i-th node
        # group_table = {}
        # for i, node in enumerate(gdef.node):
        #     if node.name.startswith("GradientDescent") or node.name.startswith("gradients"):
        #         prefix = '/'.join(node.name.split('/')[1:3])
        #     else:
        #         prefix = '/'.join(node.name.split('/')[:2])
        #     if prefix in group_table:
        #         group_table[prefix].append(i)
        #     else:
        #         group_table[prefix] = [i]
        # return list(group_table.values())

        from utils import group_around_topk_costs
        from tge import TGE

        base_groups = TGE(gdef, [dev for dev, _, _ in devices]).get_groups()
        id_list = group_around_topk_costs(gdef, base_groups, prof_data[devices[0][1]], n_groups-1) # TODO: use average time in all gpu types? weighted average?
        return list(groupby(enumerate(id_list), key=cadr, value=car).values())

    n_groups = 2 * len(devices)
    op_groups = group_with_topk_layers(n_groups)

    parameter_sizes = np.zeros(n_groups)
    tensor_sizes = np.zeros((n_groups, n_groups))
    computation_times = np.zeros((n_groups, len(devices), 4))

    name_dict = { node.name: i for i, node in enumerate(gdef.node) }
    group_dict = { nodeid: groupid for groupid, nodes in enumerate(op_groups) for nodeid in nodes }
    for thisnodeid, node in enumerate(gdef.node):
        thisgroupid = group_dict[i]
        for input in node.input:
            x, input_index = parse_input(input)
            nodeid = name_dict[x]
            groupid = group_dict[nodeid]
            tensorsize = get_input_size(gdef.node[nodeid], input_index, batchsize)
            tensor_sizes[(thisgroupid, groupid)] += tensorsize / 100_000_000
        for devid, (_, gtype, _) in enumerate(devices):
            computation_times[thisgroupid, devid, 0] += prof_data[gtype][(node.name, 1)][0] / 10_000
            computation_times[thisgroupid, devid, 1] += prof_data[gtype][(node.name, 2)][0] / 10_000
            computation_times[thisgroupid, devid, 2] += prof_data[gtype][(node.name, 4)][0] / 10_000
            computation_times[thisgroupid, devid, 3] += prof_data[gtype][(node.name, 8)][0] / 10_000
        # TODO: identify parameter size

    op_feats = [[np.mean(computation_times[i, :, x]) for x in range(4)] + parameter_sizes[i] + tensor_sizes[i, i] for i in range(n_groups)]
    tensor_feats = []
    place_feats = []
    edge_prev = ([], [])
    edge_succ = ([], [])
    edge_place = ([], [])
    edge_serve = ([], [])

    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                continue
            if tensor_sizes[i, j] > 0:
                edge_prev[0].append(i)
                edge_prev[1].append(j)
                edge_succ[0].append(j)
                edge_succ[1].append(i)
                tensor_feats.append([tensor_sizes[i, j]])

    for i in range(n_groups):
        for j, (_, gtype, _) in enumerate(devices):
            edge_place[0].append(i)
            edge_place[1].append(j)
            edge_serve[0].append(j)
            edge_serve[1].append(i)
            place_feats.append([ computation_times[i, j, x] for x in range(4) ])

    prof_data_combined = { key: [0 for _ in devices] for key, times in prof_data[devices[0][1]].items() }
    for i, (_, gtype, _) in enumerate(devices):
        for key, times in prof_data[gtype].items():
            prof_data_combined[key][i] = times[0]

    g = dgl.heterograph({
        ('device', 'link', 'device'): edge_link,
        ('op', 'prev', 'op'): edge_prev,
        ('op', 'succ', 'op'): edge_succ,
        ('op', 'place', 'device'): edge_place,
        ('device', 'serve', 'op'): edge_serve
    })

    return {
        "graph": g,
        "gdef": gdef,
        "prof_data": prof_data_combined,
        "devices": devices,
        "op_groups": op_groups,
        "op_feats": op_feats,
        "device_feats": device_feats,
        "tensor_feats": tensor_feats,
        "place_feats": place_feats,
        "link_feats": link_feats,
        "batchsize": batchsize,
        # the two are workarounds; should write a graph parser in tge.py to get the links and bandwidth from graph
        "inter": inter,
        "intra": intra,
        "nccl_models": nccl_models
    }

def get_all_data():
    models = []
    for m in ("vgg", "resnet", "inception", "transformer", "bert"): # "mobilenet", "nasnet"
        agg_prof_data = {}
        gdef, batchsize = None, None
        for gtype in ('1080ti', 'v100'):
            gdef, prof_data, _, batchsize = load("{}_{}.pickle".format(m, gtype))
            agg_prof_data[gtype] = prof_data
        models.append((gdef, agg_prof_data, batchsize))

    # topos1 = [gen_topo([
    #     ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:2", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:3", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:4", 2, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:5", 2, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:6", 2, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:7", 2, 6<<30),
    # ], intra=bandwidth) for bandwidth in (4000, 40000)]
    # topos2 = [gen_topo([
    #     ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:2", 2, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:3", 2, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:4", 3, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:5", 3, 6<<30),
    # ], intra=bandwidth) for bandwidth in (4000, 20000, 80000)]
    # topos3 = [gen_topo([
    #     ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:2", 1, 6<<30),
    #     ("/job:worker/replica:0/task:0/device:GPU:3", 1, 6<<30),
    #     ("/job:worker/replica:0/task:1/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:1/device:GPU:1", 1, 6<<30),
    # ], intra=bandwidth, inter=1000) for bandwidth in (8000, 80000)]
    # topos4 = [gen_topo([
    #     ("/job:worker/replica:0/task:0/device:GPU:0", 2, 6<<30),
    #     ("/job:worker/replica:0/task:1/device:GPU:0", 1, 10<<30),
    #     ("/job:worker/replica:0/task:1/device:GPU:1", 1, 10<<30),
    # ], intra=8000, inter=2810)]
    # topos5 = [gen_topo([
    #     ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:1/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:1/device:GPU:1", 1, 6<<30),
    #     ("/job:worker/replica:0/task:2/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:2/device:GPU:1", 1, 6<<30),
    #     ("/job:worker/replica:0/task:2/device:GPU:2", 1, 6<<30),
    #     ("/job:worker/replica:0/task:3/device:GPU:0", 1, 6<<30),
    #     ("/job:worker/replica:0/task:3/device:GPU:1", 1, 6<<30),
    #     ("/job:worker/replica:0/task:3/device:GPU:2", 1, 6<<30),
    #     ("/job:worker/replica:0/task:3/device:GPU:3", 1, 6<<30),
    # ], intra=8000, inter=2810)]
    topos6 = [([
        ("/job:worker/replica:0/task:0/device:GPU:0", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:1/device:GPU:0", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:1/device:GPU:1", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:2/device:GPU:0", 'v100', 8<<30),
        ("/job:worker/replica:0/task:2/device:GPU:1", 'v100', 8<<30),
        ("/job:worker/replica:0/task:2/device:GPU:2", 'v100', 8<<30),
        ("/job:worker/replica:0/task:2/device:GPU:3", 'v100', 8<<30),
    ], 8000, 2810)]
    topos7 = [([
        ("/job:worker/replica:0/task:0/device:GPU:0", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:1/device:GPU:0", '1080ti', 6<<30),
        ("/job:worker/replica:0/task:2/device:GPU:0", 'v100', 8<<30),
        ("/job:worker/replica:0/task:2/device:GPU:1", 'v100', 8<<30),
    ], 8000, 2810)]

    return [gen_data(gdef, prof_data, batchsize, devices, intra, inter) for gdef, prof_data, batchsize in models for devices, intra, inter in topos6 + topos7]

# prim's algorithm
# alternative: https://networkx.github.io/documentation/stable/reference/algorithms/tree.html#module-networkx.algorithms.tree.mst
def k_spanning_tree(g, weights, k, seed=0):
    def get_weight(center, neighbor):
        return weights[ng.adj[center][neighbor][0]['id']]

    ng = g.to_networkx()
    tree_nodes = [seed]
    tree_edges = []
    while True:
        bridges = [(center, neighbor) for center in tree_nodes for neighbor in ng.adj[center] if neighbor not in tree_nodes ]
        if len(bridges) == 0:
            break
        highest_weight = np.max([ get_weight(center, neighbor) for center, neighbor in bridges ])
        index_of_edge_to_add = np.random.choice([ i for i, (center, neighbor) in enumerate(bridges) if get_weight(center, neighbor) == highest_weight ])
        center, neighbor = bridges[index_of_edge_to_add]
        tree_nodes.append(neighbor)
        tree_edges.append((center, neighbor, highest_weight))
    tree_edges.sort(key=lambda x: x[2])
    tree_edges = set( (center, neighbor) for center, neighbor, weight in tree_edges[k-1:] )
    groups = []
    for node in tree_nodes:
        for group in groups:
            for neighbor in group:
                if (node, neighbor) in tree_edges or (neighbor, node) in tree_edges:
                    group.append(node)
                    break
            else:
                continue
            break
        else:
            groups.append([node])

    return groups

def parse_input(input):
    if input[0] == '^':
        node = input[1:]
        input_index = 0
    else:
        node = input.split(':')[0]
        try:
            input_index = int(input.split(':')[1])
        except:
            input_index = 0
    return node, input_index

def get_input_size(nodedef, input_index, batchsize):
    try:
        shape = [ dim.size for dim in nodedef.attr["_output_shapes"].list.shape[input_index].dim ]
        if len(shape) > 0 and shape[0] == -1:
            shape[0] = batchsize
        tensorsize = 1
        for size in shape:
            if size == -1:
                tensorsize = 0
                break
            tensorsize *= size
        return tensorsize
    except:
        return 0
