import dgl
import re
import numpy as np
import pickle
import math
import itertools
import networkx as nx
import os
import sys
sys.path.append("../")
sys.path.append("../../")
from utils import groupby, car, cadr, cdr, info, load, save
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
import protobuf.hlo_execution_profile_data_pb2 as profiler_pb2



opcode_index_dict = load("opcode_index_dict.pkl")




def gen_single_computation_data(computation_def,exe_time):
    # edges
    instruction_edge_prev = [], []
    instruction_edge_succ = [], []
    computation_to_final = [], []

    # dicts
    id_instruction_dict = {}
    instructionName_index_dict = {}
    instructionName_instruction_dict = {}
    id_computation_dict = {}
    computationName_index_dict = {}
    computationName_computation = {}



    tensor_sizes = np.zeros((len(id_instruction_dict), len(id_instruction_dict)))
    instruction_edge_feats = []
    call_computation_edge_feats = []
    in_computation_edge_feats = []
    to_final_edge_feats = []

    instruction_feats = []  # opcode,oprand number, output_size
    computation_feats = []  # instructions number, parameters number
    final_feats = []  # nums of computations

    final_feats.append([len(computation_def.instructions)])

    for instruction in computation_def.instructions:
        to_final_edge_feats.append([1])

        output_size = get_output_size(instruction)
        in_computation_edge_feats.append([output_size])  # in_computation_edge_features
        instruction_feats.append(one_hot_opcode(instruction.opcode) + [len(instruction.operand_ids)] + [
            output_size])  # instruction features

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
            tensor_sizes[instructionName_index_dict[operand_instruction.name], instructionName_index_dict[
                instruction.name]] = size
            instruction_edge_feats.append([size])


    g = dgl.heterograph({
        ('instruction', 'prev', 'instruction'): instruction_edge_prev,
        ('instruction', 'succ', 'instruction'): instruction_edge_succ,
        ('instruction', 'to_final_edge_feats', 'final'): computation_to_final

    })

    return {"graph": g,
            "instruction_feats": instruction_feats,
            "final_feats": final_feats,
            "instruction_edge_feats": instruction_edge_feats,
            "to_final_edge_feats": to_final_edge_feats,
            "execution_time":exe_time
            }


def gen_data_from_hlo_def(hlo_def,profile_def):
    datasets = []
    ComputationName_Time_Dict = dict()

    printer_data = profile_def.printer_data
    profiler_counters =profile_def.profile_counters
    for computation_info in printer_data.computation_infos:
        ComputationName_Time_Dict[computation_info.name] = (profiler_counters[computation_info.profile_index]/1.6325)/(10**6)    #ms




    for computation in (hlo_def.computations):
        ret = gen_single_computation_data(computation,ComputationName_Time_Dict[computation.name])
        datasets.append(ret)
    return datasets







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




def get_train_single_data():
    path_base = "/home/net/xiaodong/trax/trax/hlo_module/{}/{}_op_fusion_level_{}_tensor_fusion_threshold_{}/{}"

    module_name = "training.module_0061.profile_hlo_module.hlo.pb"
    profiler_name = "hlo_execution_profile_data_61"
    worker_num  = 6
    model_name = "transformer"
    op_levels  = [1111]
    tensor_thresholds = [0]
    training_datas = []
    for op_level in op_levels:
        for tensor_threshold in tensor_thresholds:
            hlo_module_path = path_base.format(worker_num,model_name,op_level,tensor_threshold,module_name)
            profiler_path = path_base.format(worker_num,model_name,op_level,tensor_threshold,profiler_name)

            if os.path.exists(hlo_module_path) and os.path.exists(profiler_path):
                with open(profiler_path, "rb") as f:
                    profiledata = profiler_pb2.HloExecutionProfileData()
                    profiledata.ParseFromString(f.read())
                with open(hlo_module_path, "rb") as f:
                    hlo_proto = hlo_pb2.HloProto()
                    hlo_proto.ParseFromString(f.read())
                    hlo_module = hlo_proto.hlo_module
                res = gen_data_from_hlo_def(hlo_module,profiledata)

                training_datas.extend(res)
    print("training data length:",len(training_datas))
    return training_datas




def get_test_single_data():
    path_base = "/home/net/xiaodong/trax/trax/hlo_module/{}/{}_op_fusion_level_{}_tensor_fusion_threshold_{}/{}"

    module_name = "training.module_0061.profile_hlo_module.hlo.pb"
    profiler_name = "hlo_execution_profile_data_61"
    worker_num  = 6
    model_name = "transformer"
    op_levels  = [1111]
    tensor_thresholds = [0]
    training_datas = []
    for op_level in op_levels:
        for tensor_threshold in tensor_thresholds:
            hlo_module_path = path_base.format(worker_num,model_name,op_level,tensor_threshold,module_name)
            profiler_path = path_base.format(worker_num,model_name,op_level,tensor_threshold,profiler_name)

            if os.path.exists(hlo_module_path) and os.path.exists(profiler_path):
                with open(profiler_path, "rb") as f:
                    profiledata = profiler_pb2.HloExecutionProfileData()
                    profiledata.ParseFromString(f.read())
                with open(hlo_module_path, "rb") as f:
                    hlo_proto = hlo_pb2.HloProto()
                    hlo_proto.ParseFromString(f.read())
                    hlo_module = hlo_proto.hlo_module
                res = gen_data_from_hlo_def(hlo_module,profiledata)

                training_datas.extend(res)
    print("training data length:",len(training_datas))
    return training_datas


if __name__=="__main__":
    files = ["3_60_after_optimizations.hlo"]
    for file in files:
        with open(file+".pb","rb") as f:
            hlo_module = hlo_pb2.HloProto()
            hlo_module.ParseFromString(f.read())
            res = gen_data(hlo_module.hlo_module)
            print(res)
