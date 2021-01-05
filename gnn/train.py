import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
import sys
sys.path.append("../")
from utils import save, load, info




def compare(pred,real):
    sort_pred = sorted(pred)
    sort_real = sorted(real)
    pred_index = [sort_pred.index(item) for item in pred]
    real_index = [sort_real.index(item) for item in real]
    info("real rank index:",real_index)
    info("pred rank index:",pred_index)
    return str(pred_index)==str(real_index)

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

with tf.device("/gpu:0"):
    model = Model()

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.Adam(learning_rate=.00004, clipnorm=6)
    L2_regularization_factor = .00001
    sample_size = 3

    for epoch in range(20000000):

        inputs = []
        graphs = []
        record_ids=[]
        execution_times=[]
        while len(record_ids)<sample_size:
            record_id = np.random.randint(len(records))
            if record_id not in record_ids:
                record_ids.append(record_id)

        for i in range(sample_size):     # random sample 3 records
            record_id = record_ids[i]
            record = records[record_id]
            instruction_feats     = tf.convert_to_tensor(record["instruction_feats"], dtype=tf.float32)
            computation_feats = tf.convert_to_tensor(record["computation_feats"], dtype=tf.float32)
            final_feats = tf.convert_to_tensor(record["final_feats"], dtype=tf.float32)
            instruction_edge_feats = tf.convert_to_tensor(record["instruction_edge_feats"], dtype=tf.float32)
            call_computation_edge_feats   = tf.convert_to_tensor(record["call_computation_edge_feats"], dtype=tf.float32)
            in_computation_edge_feats  = tf.convert_to_tensor(record["in_computation_edge_feats"], dtype=tf.float32)
            to_final_edge_feats  = tf.convert_to_tensor(record["to_final_edge_feats"], dtype=tf.float32)

            inputs.append([instruction_feats, computation_feats,final_feats, instruction_edge_feats, call_computation_edge_feats, in_computation_edge_feats,to_final_edge_feats])
            graphs.append(record["graph"])
            execution_times.append(record["execution_time"])



        # learn
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            ranks = []
            for i in range(sample_size):

                model.set_graph(graphs[i])
                ranklogit = model(inputs[i], training=True)
                ranklogit = tf.math.reduce_mean(ranklogit)
                ranks.append(ranklogit)

            loss = 0
            for i in range(len(ranks)):
                for j in range(len(ranks)):
                    #loss += tf.cond(execution_times[i] > execution_times[j], lambda: tf.math.log(1+tf.math.exp(ranks[j]-ranks[i])), lambda: 0)
                    loss += tf.cond(execution_times[i] > execution_times[j], lambda: tf.math.reduce_logsumexp([0, ranks[j] - ranks[i]]), lambda: 0)

            loss = tf.dtypes.cast(loss, tf.float32)
            info("rank loss:",loss.numpy())
            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)
            info("real_time:",execution_times)
            rank_numpy = [rank.numpy() for rank in ranks]
            info("predict_rank:",rank_numpy)
            info(record_ids, loss.numpy())
            if compare(rank_numpy,execution_times):
                info("prediction success!")
            else:
                info("prediction fail!")



            grads = tape.gradient(loss, model.trainable_weights)
            #info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # checkpoint
        if epoch % 50 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")

