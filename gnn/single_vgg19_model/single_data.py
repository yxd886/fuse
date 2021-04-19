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
    instruction_to_final = [], []

    # dicts
    id_instruction_dict = {}
    instructionName_index_dict = {}



    instruction_edge_feats = []
    in_computation_edge_feats = []
    to_final_edge_feats = []

    instruction_feats = []  # opcode,oprand number, output_size
    final_feats = []  # nums of computations

    final_feats.append([len(computation_def.instructions)])

    for index,instruction in enumerate(computation_def.instructions):
        id_instruction_dict[instruction.id] = instruction
        instructionName_index_dict[instruction.name] = index

        instruction_to_final[0].append(index)
        instruction_to_final[1].append(0)


        to_final_edge_feats.append([1])

        output_size = get_output_size(instruction)
        in_computation_edge_feats.append([output_size])  # in_computation_edge_features
        instruction_feats.append(one_hot_opcode(instruction.opcode) + [len(instruction.operand_ids)] + [
            output_size])  # instruction features

        # find oprand of this instruction
        for operand_id in instruction.operand_ids:
            operand_instruction = id_instruction_dict[operand_id]
            instruction_edge_prev[0].append(instructionName_index_dict[instruction.name])
            instruction_edge_prev[1].append(instructionName_index_dict[operand_instruction.name])
            instruction_edge_succ[0].append(instructionName_index_dict[operand_instruction.name])
            instruction_edge_succ[1].append(instructionName_index_dict[instruction.name])

            size = get_output_size(operand_instruction)
            instruction_edge_feats.append([size])


    g = dgl.heterograph({
        ('instruction', 'prev', 'instruction'): instruction_edge_prev,
        ('instruction', 'succ', 'instruction'): instruction_edge_succ,
        ('instruction', 'to_final_edge_feats', 'final'): instruction_to_final

    })

    return {"graph": g,
            "instruction_feats": instruction_feats,
            "final_feats": final_feats,
            "instruction_edge_feats": instruction_edge_feats,
            "to_final_edge_feats": to_final_edge_feats,
            "execution_time":exe_time
            }



def get_cost_model(profiler_path="training_hlo_execution_profile_data",hlo_module_path="training.hlo.pb"):

    try:
        cost_model = load("cost_model.pkl")
    except:
        cost_model = {}
        cost_model["InstructionName_Time_Dict"] = {}
        cost_model["InstructionName_Time_Dict"] = {}
    InstructionName_Time_Dict = cost_model["InstructionName_Time_Dict"]
    Tuple_Time_Dict = cost_model["InstructionName_Time_Dict"]


    if os.path.exists(hlo_module_path) and os.path.exists(profiler_path):
        with open(profiler_path, "rb") as f:
            profile_def = profiler_pb2.HloExecutionProfileData()
            profile_def.ParseFromString(f.read())
        with open(hlo_module_path, "rb") as f:
            hlo_proto = hlo_pb2.HloProto()
            hlo_proto.ParseFromString(f.read())
            hlo_module = hlo_proto.hlo_module

    Name_Instruction_Dict = {}
    for computation in (hlo_module.computations):
        for instruction in computation.instructions:
            Name_Instruction_Dict[instruction.name] = instruction



    printer_data = profile_def.printer_data
    profiler_counters =profile_def.profile_counters
    for computation_info in printer_data.computation_infos:
        for instruction_info in computation_info.instruction_infos:
            instruction_name = instruction_info.short_name.split(" ")[0].strip()[1:]

            exe_time = (profiler_counters[instruction_info.profile_index]/1.6325)/1000   #us
            #using name as key
            InstructionName_Time_Dict[instruction_name] = exe_time
            #using op type and shape tuple as key
            instruction = Name_Instruction_Dict[instruction_name]
            opcode = instruction.opcode
            shape =get_shape_string(instruction)
            Tuple_Time_Dict[(opcode,shape)] = exe_time

    save(cost_model,"cost_model.pkl")
    return InstructionName_Time_Dict,Tuple_Time_Dict,hlo_module


def gen_data_from_hlo_def(hlo_def,profile_def):
    datasets = []
    InstructionName_Time_Dict = dict()
    Id_Computation_dict = {}
    fusion_instructionName_computation_dict = {}
    for computation in (hlo_def.computations):
        Id_Computation_dict[computation.id]=computation
    for computation in (hlo_def.computations):
        for instruction in computation.instructions:
            if instruction.opcode=="fusion":
                assert(len(instruction.called_computation_ids)==1)
                fusion_instructionName_computation_dict[instruction.name] = Id_Computation_dict[instruction.called_computation_ids[0]]

    printer_data = profile_def.printer_data
    profiler_counters =profile_def.profile_counters
    for computation_info in printer_data.computation_infos:
        for instruction_info in computation_info.instruction_infos:
            instruction_name = instruction_info.short_name.split(" ")[0].strip()[1:]
            InstructionName_Time_Dict[instruction_name] = (profiler_counters[instruction_info.profile_index]/1.6325)/1000   #us


    for fusion_instructionName in fusion_instructionName_computation_dict:
        computation = fusion_instructionName_computation_dict[fusion_instructionName]
        ret = gen_single_computation_data(computation,InstructionName_Time_Dict[fusion_instructionName])
        datasets.append(ret)
    return datasets




def get_shape_string(instruction):
    return str(instruction.shape)


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

    module_name = "training.hlo.pb"
    profiler_name = "training_hlo_execution_profile_data"
    worker_num  = 6
    model_name = "transformer"
    training_datas = []

    hlo_module_path = module_name
    profiler_path = profiler_name

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
    return get_train_single_data()

