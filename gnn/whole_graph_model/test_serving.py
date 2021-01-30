import tensorflow.compiler.xla.service.hlo_pb2 as hlo_pb2
import requests
import os

url_base = "localhost:3335"

op_level = 0
threshold =0

path = "test.pb"
if os.path.exists(path) and os.path.exists(path):
    with open(path, "rb") as f:
        data = {
            "data": f.read()
        }

        rep = requests.post("http://" + url_base + "/predict", data=data)

        print(rep.json())