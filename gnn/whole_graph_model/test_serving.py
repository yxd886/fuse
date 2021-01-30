import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
import requests
import os

url_base = "localhost:3335"

op_level = 0
threshold =0

path = "test_hlo.pb"
if os.path.exists(path) and os.path.exists(path):
    with open(path, "rb") as f:
        hlo_module = hlo_pb2.HloProto()
        hlo_module.ParseFromString(f.read())

        header = {"Content-Type": "image/gif"}
        data = hlo_module.SerializeToString()


        rep = requests.post("http://" + url_base + "/predict", data=data,headers=header)

        print(rep.json())