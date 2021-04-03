
import sys
sys.path.append("../")
sys.path.append("../../")

import tensorflow as tf
from single_data import get_cost_model,gen_single_computation_data,get_test_single_data,get_train_single_data
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
from single_model import SingleModel
from utils import save, load, info



class CostModel():
    def __init__(self,profiler_path="training_hlo_execution_profile_data",hlo_module_path="training.hlo.pb"):
        self.name_time_dict,self.tuple_time_dict,self.init_hlo_module = get_cost_model(profiler_path,hlo_module_path)

        self.cache = {}

        with tf.device("/gpu:0"):
            self.model = SingleModel()
            try:
                self.model.load_weights('weights')
                print("load saved weight")
            except:
                print("no saved weight")

            try:
                self.records = load("single_records")
                info("load saved records")
            except:
                self.records = get_train_single_data()
                info("no saved records")
            try:
                self.tests = load("single_tests")
                info("load saved tests")
            except:
                self.tests = get_test_single_data()
                info("no saved tests")
                save(self.tests,"single_tests")

    def estimate_instruction_time(self,instruction):
        opcode = instruction.opcode
        shape = instruction.shape
        if  (opcode,str(shape)) not in self.tuple_time_dict:
            print((opcode,str(shape)))
            return 0
        return self.tuple_time_dict[(opcode,str(shape))]


    def test_accuracy(self,hlo_model=None):
        if hlo_model==None:
            hlo_model = self.init_hlo_module
        else:
            with open(hlo_model, "rb") as f:
                hlo_proto = hlo_pb2.HloProto()
                hlo_proto.ParseFromString(f.read())
                hlo_module = hlo_proto.hlo_module
        estimated_time = self.estimate_time(hlo_model)
        print("estimated_time withgnn:",estimated_time)

        estimated_time,all_reduce_time = self.estimate_time_without_gnn(self.init_hlo_module)
        print("estimated_time without gnn:",estimated_time)
        print("all-reduce_time without gnn:",all_reduce_time)

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

    def acquire_gnn_by_record(self,record):
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

    def estimate_time_without_gnn(self,hlo_module):
        entry_computation = None

        id_computation_dict = {}
        for computation in hlo_module.computations:
            if computation.id ==hlo_module.entry_computation_id:
                entry_computation = computation
            id_computation_dict[computation.id] = computation
        assert (entry_computation)
        time = 0
        all_reduce_time = 0
        for instruction in entry_computation.instructions:
            time+=self.name_time_dict[instruction.name]
            if instruction.opcode=="all-reduce":
                all_reduce_time+=self.name_time_dict[instruction.name]
        return time,all_reduce_time

    def get_fuse_key(self,fuse_computation):
        key = []
        for instruction in fuse_computation.instructions:
            key.append((instruction.opcode,instruction.shape))
        return str(key)

    def estimate_time(self,hlo_module):
        entry_computation = None
        id_computation_dict = {}
        for computation in hlo_module.computations:
            if computation.id ==hlo_module.entry_computation_id:
                entry_computation = computation
            id_computation_dict[computation.id] = computation
        assert (entry_computation)

        time = 0
        all_reduce_time = 0
        fused_counter = 0
        for instruction in entry_computation.instructions:
            if instruction.opcode!="fusion":
                instruction_time = self.estimate_instruction_time(instruction)
                time+=instruction_time
                if instruction.opcode == "all-reduce":
                    all_reduce_time += instruction_time
            else:
                fused_counter = fused_counter+1
                computation = id_computation_dict[instruction.called_computation_ids[0]]
                key = self.get_fuse_key(computation)
                if key in self.cache:
                    time+=self.cache[key]
                else:
                    current_time = self.acquire_gnn(computation)
                    self.cache[key] = current_time
                    time+=current_time
        print("total time:",time, "all-reduce time:",all_reduce_time,"fused number:",fused_counter)
        return time


    def draw_picture(self):
        import matplotlib.pyplot as plt
        import numpy as np
        time_tuples = []
        for record in self.records:
            real_time = record["execution_time"]
            estimate_time = self.acquire_gnn_by_record(record)
            time_tuples.append((real_time,estimate_time))
        time_tuples.sort(key=lambda item: item[0])
        x = np.arange(len(time_tuples))
        real_y = np.array([item[0] for item in time_tuples ])
        estimated_y = np.array([item[1] for item in time_tuples ])
        plt.plot(x, real_y,color="r",label="real execution time")
        plt.plot(x, estimated_y,color="b",label="estimated time")
        plt.savefig("train_estimate.png")

        plt.clf()

        time_tuples = []
        for test in self.tests:
            real_time = test["execution_time"]
            estimate_time = self.acquire_gnn_by_record(test)
            time_tuples.append((real_time,estimate_time))
        time_tuples.sort(key=lambda item: item[0])
        x = np.arange(len(time_tuples))
        real_y = np.array([item[0] for item in time_tuples ])
        estimated_y = np.array([item[1] for item in time_tuples ])
        plt.plot(x, real_y,color="r",label="real execution time")
        plt.plot(x, estimated_y,color="b",label="estimated time")
        plt.savefig("test_estimate.png")





