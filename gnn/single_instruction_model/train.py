import numpy as np
import time
import tensorflow as tf


import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../protobuf/")
from single_data import get_train_single_data,get_test_single_data
from single_model import SingleModel
from utils import save, load, info




def compare(pred,real):
    a = pred-real
    c= a/real
    print("predict:",pred)
    print("real:",real)
    print("individual predict ratio:",c)
    print("total ratio:",sum(c)/len(c))

try:
    records = load("single_records")
    info("load saved records")
except:
    records = get_train_single_data()
    info("no saved records")
    save(records, "single_records")

records = records+records+records


try:
    tests = load("single_tests")
    info("load saved tests")
except:
    tests = get_test_single_data()
    info("no saved tests")
    save(tests, "single_tests")

tests = tests+tests+tests


with tf.device("/gpu:0"):
    model = SingleModel()

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.Adam(learning_rate=.00004, clipnorm=6)
    L2_regularization_factor = .0001
    sample_size = 32
    train = True
    if train==False:
        sample_size = 2

    test_counter = 0

    max_train_iteration = len(records)//sample_size
    max_test_iteration = len(tests)//sample_size


    for epoch in range(20000000):
        if train:
            inputs = []
            graphs = []
            record_ids=[]
            execution_times=[]
            #while len(record_ids)<sample_size:
            #    record_id = np.random.randint(len(records))
            #    if record_id not in record_ids:
            #        record_ids.append(record_id)
            real_epoch = epoch%max_train_iteration
            record_ids = range(sample_size*real_epoch,(real_epoch+1)*sample_size)

            for i in range(sample_size):     # random sample 3 records
                record_id = record_ids[i]
                record = records[record_id]
                instruction_feats     = tf.convert_to_tensor(record["instruction_feats"], dtype=tf.float32)
                final_feats = tf.convert_to_tensor(record["final_feats"], dtype=tf.float32)
                instruction_edge_feats = tf.convert_to_tensor(record["instruction_edge_feats"], dtype=tf.float32)
                to_final_edge_feats  = tf.convert_to_tensor(record["to_final_edge_feats"], dtype=tf.float32)

                inputs.append([instruction_feats,final_feats, instruction_edge_feats,to_final_edge_feats])
                graphs.append(record["graph"])
                execution_times.append(record["execution_time"])


            #learn
            with tf.GradientTape() as tape:
                tape.watch(model.trainable_weights)
                ranks = []
                for i in range(sample_size):

                    model.set_graph(graphs[i])
                    ranklogit = model(inputs[i], training=True)
                    ranklogit = tf.math.reduce_mean(ranklogit)
                    ranks.append(ranklogit)

                loss = 0
                for k,rank in enumerate(ranks):
                    loss = loss+tf.math.square(rank-execution_times[k])
                loss = loss/len(ranks)
                info("rank loss:",loss.numpy())
                if L2_regularization_factor > 0:
                    for weight in model.trainable_weights:
                        loss += L2_regularization_factor * tf.nn.l2_loss(weight)
                rank_numpy = np.array([rank.numpy() for rank in ranks])
                info(record_ids, loss.numpy())
                compare(rank_numpy,np.array(execution_times))

                grads = tape.gradient(loss, model.trainable_weights)
                #info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # checkpoint
            if epoch % 50 == 0:
                info("==== save ====")
                model.save_weights('weights')
                #save(records, "records")
        else: # test
            inputs = []
            graphs = []
            test_ids = []
            execution_times = []
            #while len(test_ids) < sample_size:
            #    test_id = np.random.randint(len(tests))
            #    if test_id not in test_ids:
            #        test_ids.append(test_id)

            real_epoch = epoch%max_test_iteration
            test_ids = range(sample_size*real_epoch,(real_epoch+1)*sample_size)

            for i in range(sample_size):  # random sample 3 records
                test_id = test_ids[i]
                test = tests[test_id]
                instruction_feats = tf.convert_to_tensor(test["instruction_feats"], dtype=tf.float32)
                computation_feats = tf.convert_to_tensor(test["computation_feats"], dtype=tf.float32)
                final_feats = tf.convert_to_tensor(test["final_feats"], dtype=tf.float32)
                instruction_edge_feats = tf.convert_to_tensor(test["instruction_edge_feats"], dtype=tf.float32)
                call_computation_edge_feats = tf.convert_to_tensor(test["call_computation_edge_feats"],
                                                                   dtype=tf.float32)
                in_computation_edge_feats = tf.convert_to_tensor(test["in_computation_edge_feats"], dtype=tf.float32)
                to_final_edge_feats = tf.convert_to_tensor(test["to_final_edge_feats"], dtype=tf.float32)

                inputs.append([instruction_feats, computation_feats, final_feats, instruction_edge_feats,
                               call_computation_edge_feats, in_computation_edge_feats, to_final_edge_feats])
                graphs.append(test["graph"])
                execution_times.append(test["execution_time"])

            # test

            ranks = []
            for i in range(sample_size):
                model.set_graph(graphs[i])
                ranklogit = model(inputs[i], training=True)
                ranklogit = tf.math.reduce_mean(ranklogit)
                ranks.append(ranklogit)

            loss = 0
            for k, rank in enumerate(ranks):
                loss = loss + tf.math.square(rank - execution_times[k])
            loss = loss / len(ranks)
            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)
            rank_numpy = np.array([rank.numpy() for rank in ranks])
            info("chosen sample index:",test_ids)
            info("loss:", loss.numpy())
            compare(rank_numpy, np.array(execution_times))

