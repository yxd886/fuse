import numpy as np
import time
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver

from utils import save, load, info
from tge import TGE

import os
os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["10.28.1.24:3806", "10.28.1.23:3901", "10.28.1.26:3901"] }, "task": {"type": "worker", "index": 0} }'

def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)

setup_workers(["10.28.1.24:3806", "10.28.1.23:3901", "10.28.1.26:3901"])

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1",
)
resolver = TFConfigClusterResolver()
cluster = resolver.cluster_spec()
dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
config = dist.update_config_proto(tf.ConfigProto())
config.ClearField("device_filters")
server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc", config=config)

gdef, _, _, batchsize = load("inception_1080ti.pickle")
strategy = load("strategy.pickle")
# strategy = { node.name: [1] + [1] * len(devices) for node in gdef.node }

for k, v in strategy.items():
    if np.sum(v[1:]) == 0:
        v[1] = 1

for k, v in strategy.items():
    v[0] = 0

tge = TGE(gdef, devices, sinks=["Adam"])
tge.set_strategy(strategy)
tge.replace_placeholder(batchsize)
tge.compile()
g = tge.get_result()

tf.import_graph_def(g)
graph = tf.get_default_graph()

opt = graph.get_operation_by_name("import/Adam/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")

sess = tf.Session(server.target, config=config)
sess.run(init)
sess.run(opt)

run_meta = tf.compat.v1.RunMetadata()
run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
sess.run(opt)

tl = timeline.Timeline(run_meta.step_stats)
with open("timeline.json", "w") as fo:
    fo.write(tl.generate_chrome_trace_format())

tic = time.perf_counter()
for _ in range(10):
    sess.run(opt)
toc = time.perf_counter()

print("time: {}".format((toc - tic) / 10))
