
import sys
sys.path.append("../")
sys.path.append("../../")

import tensorflow as tf
from single_data import get_cost_model,gen_single_computation_data
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
from single_model import SingleModel



class CostModel():
    def __init__(self,profiler_path="hlo_execution_profile_data_61",hlo_module_path="training.module_0061.profile_hlo_module.hlo.pb"):
        self.name_time_dict,self.tuple_time_dict = get_cost_model(profiler_path,hlo_module_path)

        with tf.device("/gpu:0"):
            self.model = SingleModel()
            try:
                self.model.load_weights('weights')
                print("load saved weight")
            except:
                print("no saved weight")

    def estimate_instruction_time(self,instruction):
        opcode = instruction.opcode
        shape = instruction.shape
        return self.tuple_time_dict[(opcode,str(shape))]


    def acquire_gnn(self,computation_def):

        record = gen_single_computation_data(computation_def,0)
        instruction_feats = tf.convert_to_tensor(record["instruction_feats"], dtype=tf.float32)
        final_feats = tf.convert_to_tensor(record["final_feats"], dtype=tf.float32)
        instruction_edge_feats = tf.convert_to_tensor(record["instruction_edge_feats"], dtype=tf.float32)
        to_final_edge_feats = tf.convert_to_tensor(record["to_final_edge_feats"], dtype=tf.float32)

        my_input = [instruction_feats, final_feats, instruction_edge_feats, to_final_edge_feats]
        graph = record["graph"]
        self.model.set_graph(graph)
        estimate_time = self.model(my_input, training=False)
        estimate_time = tf.math.reduce_mean(estimate_time).numpy()
        return estimate_time

    def estimate_time(self,hlo_module):
        entry_computation = None
        id_computation_dict = {}
        for computation in hlo_module.computations:
            if computation.id ==hlo_module.entry_computation_id:
                entry_computation = computation
            id_computation_dict[computation.id] = computation
        assert (entry_computation)

        time = 0
        for instruction in entry_computation.instructions:
            if instruction.opcode!="fusion":
                time+=self.estimate_instruction_time(instruction)
            else:
                computation = id_computation_dict[instruction.called_computation_ids[0]]
                time+=self.acquire_gnn(computation)
        return time

