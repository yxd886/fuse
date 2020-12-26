import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, sample_and_evaluate, replication_number_feasibility_rounding
from search import search
from utils import save, load, info
from tge import TGE

records = load("records")

with tf.device("/gpu:1"):
    model = Model()
    model.load_weights('weights')

    for rid, record in enumerate(records):
        info(rid, record['elites'][-1][0], [ loss_env for loss_env, _, _ in record['reference'] ] )

    # raise SystemExit()

    record = records[7]

    op_feats     = tf.convert_to_tensor(record["op_feats"], dtype=tf.float32)
    device_feats = tf.convert_to_tensor(record["device_feats"], dtype=tf.float32)
    tensor_feats = tf.convert_to_tensor(record["tensor_feats"], dtype=tf.float32)
    link_feats   = tf.convert_to_tensor(record["link_feats"], dtype=tf.float32)
    place_feats  = tf.convert_to_tensor(record["place_feats"], dtype=tf.float32)
    model.set_graph(record["graph"])

    nodelogit, nccllogit = model([op_feats, device_feats, tensor_feats, link_feats, place_feats], training=False)
    nodep = tf.nn.softmax(nodelogit).numpy()
    ncclp = tf.math.sigmoid(nccllogit).numpy()
    loss_env, nodemask, ncclmask = search(record, nodep, ncclp, n_gen=35)
    nodemask = np.reshape(nodemask, (len(record['op_groups']), len(record['devices'])))

    save((nodemask, ncclmask), "shit.pickle")

    replication_number_feasibility_rounding(record, nodemask)

    # loss_env, nodemask, ncclmask = record['elites'][-1]

    info(nodemask, ncclmask)


    gdef = record["gdef"]
    strategy = { gdef.node[i].name: [int(ncclmask[gi])] + [ int(nodemask[gi, j]) for j in range(nodemask.shape[1]) ] for gi, group in enumerate(record["op_groups"]) for i in group }
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    d = {}
    for n, s in strategy.items():
        if tuple(s) not in d:
            d[tuple(s)] = 1
        else:
            d[tuple(s)] += 1
    for s, c in d.items():
        info(s, c)

    save(strategy, "strategy.pickle")

    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]], sinks=['Adam'])
    tge.set_strategy(strategy)
    tge.fill_batchsize(record['batchsize'])
    tge.replace_placeholder(record['batchsize'])
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"], "trace_best.json")

    strategy = { gdef.node[i].name: [1] + [ 1 for j in range(nodemask.shape[1]) ] for gi, group in enumerate(record["op_groups"]) for i in group }
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]], sinks=['Adam'])
    tge.set_strategy(strategy)
    tge.fill_batchsize(record['batchsize'])
    tge.replace_placeholder(record['batchsize'])
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"], "trace_dp.json")


