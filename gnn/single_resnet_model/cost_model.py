
import sys
sys.path.append("../")
sys.path.append("../../")

import tensorflow as tf
from single_data import get_cost_model,gen_single_computation_data,get_test_single_data,get_train_single_data
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
from single_model import SingleModel
from utils import save, load, info
import copy


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
            return self.name_time_dict[instruction.name]
        return self.tuple_time_dict[(opcode,str(shape))]



    def test_accuracy(self,hlo_model=None):
        if hlo_model==None:
            hlo_model = self.init_hlo_module
        else:
            with open(hlo_model, "rb") as f:
                hlo_proto = hlo_pb2.HloProto()
                hlo_proto.ParseFromString(f.read())
                hlo_model = hlo_proto.hlo_module
        estimated_time = self.estimate_time(hlo_model)
        print("estimated_time withgnn:",estimated_time)

        estimated_time,all_reduce_time = self.estimate_time_without_gnn(hlo_model)
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
            time+=self.estimate_instruction_time(instruction)
            if instruction.opcode=="all-reduce":
                all_reduce_time+=self.estimate_instruction_time(instruction)
        return time,all_reduce_time

    def get_fuse_key(self,fuse_computation):
        key = []
        for instruction in fuse_computation.instructions:
            key.append((instruction.opcode,instruction.shape))
        return str(key)


    def estimate_single_instruction(self,id_computation_dict,instruction):
        if instruction.opcode != "fusion":
            instruction_time = self.estimate_instruction_time(instruction)
            return instruction_time
        else:
            computation = id_computation_dict[instruction.called_computation_ids[0]]
            key = self.get_fuse_key(computation)
            if key in self.cache:
                return self.cache[key]
            else:
                current_time = self.acquire_gnn(computation)
                self.cache[key] = current_time
                return current_time


    def estimate_time(self,hlo_module,overlap=True):
        entry_computation = None
        id_computation_dict = {}
        for computation in hlo_module.computations:
            if computation.id ==hlo_module.entry_computation_id:
                entry_computation = computation
            id_computation_dict[computation.id] = computation
        assert (entry_computation)

        if not overlap:
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
        if overlap:
            unexecuted_instructions = list(entry_computation.instructions)
            executed_instructon_ids = []
            id_time_dict = {}
            instruction_timeline=0
            allreduce_timeline=0

            all_reduce_time = 0
            computation_time = 0

            while len(unexecuted_instructions)>0:
                current_instruction = unexecuted_instructions[0]
                operand_ids = current_instruction.operand_ids
                if self.no_dependency(operand_ids,executed_instructon_ids):
                    #compute start time
                    duration = self.estimate_single_instruction(id_computation_dict,current_instruction)
                    if current_instruction.opcode!="all-reduce":
                        computation_time+=duration
                        if len(operand_ids)==0:
                            id_time_dict[current_instruction.id] = {}
                            id_time_dict[current_instruction.id]["start_time"] = instruction_timeline
                            id_time_dict[current_instruction.id]["end_time"] = instruction_timeline+duration
                            id_time_dict[current_instruction.id]["duration"] = duration
                            instruction_timeline +=duration
                        else:
                            end_times = []
                            for id in operand_ids:
                                end_times.append(id_time_dict[id]["end_time"])
                            max_end_time = max(end_times)
                            earlist_start_time = max([max_end_time,instruction_timeline])
                            id_time_dict[current_instruction.id] = {}
                            id_time_dict[current_instruction.id]["start_time"] = earlist_start_time
                            id_time_dict[current_instruction.id]["end_time"] = earlist_start_time+duration
                            id_time_dict[current_instruction.id]["duration"] = duration
                            instruction_timeline =earlist_start_time+duration
                    else:
                        all_reduce_time+=duration
                        end_times = []
                        for id in operand_ids:
                            end_times.append(id_time_dict[id]["end_time"])
                        max_end_time = max(end_times)
                        earlist_start_time = max([max_end_time, allreduce_timeline])
                        id_time_dict[current_instruction.id] = {}
                        id_time_dict[current_instruction.id]["start_time"] = earlist_start_time
                        id_time_dict[current_instruction.id]["end_time"] = earlist_start_time + duration
                        id_time_dict[current_instruction.id]["duration"] = duration
                        allreduce_timeline = earlist_start_time + duration

                    unexecuted_instructions.remove(unexecuted_instructions[0])
                    executed_instructon_ids.append(current_instruction.id)

                else:

                    unexecuted_instructions.insert(2,copy.deepcopy(current_instruction))
                    unexecuted_instructions.remove(unexecuted_instructions[0])
            print("total time:",max([instruction_timeline,allreduce_timeline]), "all-reduce-timeline:",allreduce_timeline,"computation_timeline:",instruction_timeline,"all-reduce-aggregated-time:",all_reduce_time,"computation-aggregated-time:",computation_time)

            return max([instruction_timeline,allreduce_timeline])





    def no_dependency(self,operand_ids,executed_instruction_ids):
        for id in operand_ids:
            if id not in executed_instruction_ids:
                return False
        return True

    def draw_picture(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import pickle as pkl
        time_tuples = []
        f_size=22
        for record in self.records:
            real_time = record["execution_time"]
            estimate_time = self.acquire_gnn_by_record(record)
            time_tuples.append((real_time, estimate_time))
        time_tuples.sort(key=lambda item: item[0])

        with open("time_data.pkl","wb") as f:
            pkl.dump(time_tuples,f)

        x = np.arange(len(time_tuples))
        real_y = np.array([item[0] for item in time_tuples])
        estimated_y = np.array([item[1] for item in time_tuples])
        plt.plot(x, real_y, color="r", label="Real execution time")
        plt.plot(x, estimated_y, color="b", alpha=0.5, label="Estimated time")
        plt.legend()

        '''
        for tl in plt.get_xticklabels():
            tl.set_fontsize(f_size-2)
            tl.set_fontstyle('normal')
        '''


        plt.xlabel("Samples", fontsize=f_size+4, style='normal', color='black')
        plt.ylabel("Execution Time (us)", fontsize=f_size - 2, style='normal',
                      color='black')


        plt.savefig("train_estimate.png")



        plt.clf()

        error = np.array([np.abs(item[1] - item[0]) / item[0] for item in time_tuples])
        count, bins_count = np.histogram(error, bins=100)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        plt.plot(bins_count[1:], pdf, color="red", label="PDF")
        plt.plot(bins_count[1:], cdf, label="CDF")
        plt.legend()

        '''
        for tl in plt.get_xticklabels():
            tl.set_fontsize(f_size-2)
            tl.set_fontstyle('normal')
        '''

        plt.xlabel("Error ratio", fontsize=f_size+4, style='normal', color='black')
        #plt.ylabel("                        Per-step Execution Time (s)", fontsize=f_size - 2, style='normal',
        #              color='black')
        plt.savefig("pdf_cdfg.png")






