import re
import json
import base64
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")

import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from data import gen_data
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
from model import Model


app = Flask(__name__)

model_file = 'my_model'
with tf.device("/gpu:0"):
    model = Model()

    try:
        model.load_weights('weights')
        print("load saved weight")
    except:
        print("no saved weight")



@app.route('/predict', methods=[ 'POST'])
def launch():
    if request.method == 'POST':
        data = request.get_data()
        hlo_module = hlo_pb2.HloProto()
        hlo_module.ParseFromString(data)
        res = gen_data(hlo_module.hlo_module)
        instruction_feats = tf.convert_to_tensor(res["instruction_feats"], dtype=tf.float32)
        computation_feats = tf.convert_to_tensor(res["computation_feats"], dtype=tf.float32)
        final_feats = tf.convert_to_tensor(res["final_feats"], dtype=tf.float32)
        instruction_edge_feats = tf.convert_to_tensor(res["instruction_edge_feats"], dtype=tf.float32)
        call_computation_edge_feats = tf.convert_to_tensor(res["call_computation_edge_feats"],
                                                           dtype=tf.float32)
        in_computation_edge_feats = tf.convert_to_tensor(res["in_computation_edge_feats"], dtype=tf.float32)
        to_final_edge_feats = tf.convert_to_tensor(res["to_final_edge_feats"], dtype=tf.float32)

        input = [instruction_feats, computation_feats, final_feats, instruction_edge_feats,call_computation_edge_feats, in_computation_edge_feats, to_final_edge_feats]
        graph = res["graph"]
        model.set_graph(graph)
        ranklogit = model(input, training=False)
        ranklogit = tf.math.reduce_mean(ranklogit).numpy()

        req = {
            "code": "0000",
            "result": str(ranklogit),
        }
        return str(ranklogit)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3335)