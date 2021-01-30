import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
import requests
import os

url_base = "localhost"
path_base = "/home/net/xiaodong/trax/trax/hlo_module/6/transformer_op_fusion_level_{}_tensor_fusion_threshold_{}/{}"
module_name = "training.module_0061.after_all_reduce_combiner.hlo.pb"
op_level = 0
threshold =0

path = path_base.format(op_level,threshold,module_name)
if os.path.exists(path) and os.path.exists(path):
    with open(path, "rb") as f:
        data = {
            "data": f.read()
        }

        rep = requests.post("http://" + url_base + "/predict", data=data)

        print(rep.json())