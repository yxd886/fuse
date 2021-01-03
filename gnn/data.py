import dgl
import re
import numpy as np
import pickle
import math
import itertools
import networkx as nx
import sys
sys.path.append("../")
from utils import groupby, car, cadr, cdr, info, load, save
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2


opcode_index_dict = load("opcode_index_dict.pkl")

def gen_data(hlo_def):
    #edges
    instruction_in_computation_link = [], []
    instruction_call_computation_link = [], []
    instruction_edge_prev = [],[]
    instruction_edge_succ = [],[]

    computation_to_final = [],[]

    #dicts
    id_instruction_dict = {}
    instructionName_index_dict= {}
    instructionName_instruction_dict= {}
    id_computation_dict= {}
    computationName_index_dict= {}
    computationName_computation= {}



    instruction_index=0
    for computation_index, computation in enumerate(hlo_def.computations):
        id_computation_dict[computation.id] = computation
        computationName_index_dict[computation.name]=computation_index
        computationName_computation[computation.name] = computation

        computation_to_final[0].append(computation_index)
        computation_to_final[1].append(0)

        for instruction in (computation.instructions):
            id_instruction_dict[instruction.id] = instruction
            instructionName_index_dict[instruction.name] = instruction_index
            instructionName_instruction_dict[instruction.name]=instruction

            # complete instruction_in_computation_link
            instruction_in_computation_link[0].append(instruction_index)
            instruction_in_computation_link[1].append(computation_index)

            instruction_index+=1


    tensor_sizes = np.zeros((len(id_instruction_dict), len(id_instruction_dict)))
    instruction_edge_feats = []
    call_computation_edge_feats = []
    in_computation_edge_feats = []
    to_final_edge_feats = []

    instruction_feats = []  # opcode,oprand number, output_size
    computation_feats = []  # instructions number, parameters number
    final_feats = [] #nums of computations

    final_feats.append([hlo_def.computations])
    for computation_index, computation in enumerate(hlo_def.computations):
        computation_feats.append([len(computation.instructions)]+[len(computation.program_shape.parameters)])        #computation features
        to_final_edge_feats.append([1])

        for instruction in computation.instructions:

            output_size = get_output_size(instruction)
            in_computation_edge_feats.append([output_size])  #in_computation_edge_features
            instruction_feats.append(one_hot_opcode(instruction.opcode)+[len(instruction.operand_ids)]+[output_size])  #instruction features

            # find oprand of this instruction
            global_size = 0
            for operand_id in instruction.operand_ids:
                operand_instruction = id_instruction_dict[operand_id]
                instruction_edge_prev[0].append(instructionName_index_dict[instruction.name])
                instruction_edge_prev[1].append(instructionName_index_dict[operand_instruction.name])
                instruction_edge_succ[0].append(instructionName_index_dict[operand_instruction.name])
                instruction_edge_succ[1].append(instructionName_index_dict[instruction.name])

                size = get_output_size(operand_instruction)
                global_size = global_size + size
                tensor_sizes[instructionName_index_dict[operand_instruction.name],instructionName_index_dict[instruction.name]] = size
                instruction_edge_feats.append([size])



            for called_computation_id in instruction.called_computation_ids:
                instruction_call_computation_link[0].append(instructionName_index_dict[instruction.name])
                instruction_call_computation_link[1].append(computationName_index_dict[id_computation_dict[called_computation_id].name])
                call_computation_edge_feats.append([global_size])



    g = dgl.heterograph({
        ('instruction', 'in', 'computation'): instruction_in_computation_link,
        ('instruction', 'prev', 'instruction'): instruction_edge_prev,
        ('instruction', 'succ', 'instruction'): instruction_edge_succ,
        ('instruction', 'call', 'computation'): instruction_call_computation_link,
        ('computation', 'to', 'final'): computation_to_final

    })

    return {"graph":g,
             "instruction_feats":instruction_feats,
             "computation_feats":computation_feats,
             "final_feats":final_feats,
             "instruction_edge_feats":instruction_edge_feats,
             "call_computation_edge_feats":call_computation_edge_feats,
             "in_computation_edge_feats":in_computation_edge_feats,
             "to_final_edge_feats":to_final_edge_feats
             }

def get_output_size(instruction):
    size = 1
    # process "tuple" instruction
    if instruction.opcode == "tuple":
        for shape in instruction.shape.tuple_shapes:
            local_size = 1
            for dimension in shape.dimensions:
                local_size = local_size * dimension
            size += local_size
    # process other instructions
    else:
        for dimension in instruction.shape.dimensions:
            size = size * dimension
    return size/100000000


def one_hot_opcode(opcode):
    global opcode_index_dict
    if opcode not in opcode_index_dict:
        opcode_index_dict[opcode] = len(opcode_index_dict)
        save(opcode_index_dict,"opcode_index_dict.pkl")
    index = opcode_index_dict[opcode]
    one_hot_targets = np.eye(len(opcode_index_dict))[index]
    return one_hot_targets


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



files = ["3_60_after_optimizations.hlo"]


for file in files:
    with open(file+".pb","rb") as f:
        hlo_module = hlo_pb2.HloProto()
        hlo_module.ParseFromString(f.read())
        res = gen_data(hlo_module.hlo_module)
        print(res)
