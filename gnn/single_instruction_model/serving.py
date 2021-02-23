import re
import json
import base64
import numpy as np
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../protobuf/")

import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from cost_model import CostModel
import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
import threading

app = Flask(__name__)

my_lock = threading.Lock()
cost_model = CostModel()


@app.route('/predict', methods=['POST'])
def launch():
    if request.method == 'POST':
        data = request.get_data()
        hlo_proto = hlo_pb2.HloProto()
        hlo_proto.ParseFromString(data)
        hlo_module = hlo_proto.hlo_module

        with my_lock:
            estimated_time = cost_model.estimate_time(hlo_module)
        return str(estimated_time)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3335)