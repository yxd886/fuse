import hlo_pb2
import hlo_execution_profile_data_pb2 as profiler_pb2
from utils import groupby, car, cadr, cdr, info, load, save

files = ["0_0_before_optimizations.hlo","0_0_after_optimizations.hlo","3_0_before_optimizations.hlo","3_0_after_optimizations.hlo","3_60_before_optimizations.hlo","3_60_after_optimizations.hlo"]
files = ["3_60_after_optimizations.hlo"]
files = ["hlo_execution_profile_data_61"]


def generate_hlo_txt():
    for file in files:
        with open(file,"rb") as f:
            hlo_module = hlo_pb2.HloProto()
            hlo_module.ParseFromString(f.read())
            generate_op_index_dict(hlo_module.hlo_module)
        with open(file+".pbtxt","w") as f:
            f.write(str(hlo_module))

def generate_profile_txt():
    for file in files:
        with open(file,"rb") as f:
            profiledata = profiler_pb2.HloExecutionProfileData()
            profiledata.ParseFromString(f.read())
        print("read success")
        with open(file+".pbtxt","w") as f:
            f.write(str(profiledata))

def generate_op_index_dict(hlo_def):
    opcode_index_dict = {}
    for computation_index, computation in enumerate(hlo_def.computations):
        for instruction in (computation.instructions):
            if instruction.opcode not in opcode_index_dict:
                opcode_index_dict[instruction.opcode] = len(opcode_index_dict)
    save(opcode_index_dict,"opcode_index_dict.pkl")


generate_profile_txt()

