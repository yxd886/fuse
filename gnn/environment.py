from tge import TGE
from utils import car, cadr, cdr, info
import numpy as np
import tensorflow as tf

def sample(logit, e=0):
    p = tf.math.sigmoid(logit)
    def f(x):
        if np.random.rand() < e:
            return np.random.choice(2)
        else:
            return int(np.random.rand() < x)
    return np.vectorize(f)(p)

def evaluate(record, ncclmask, nodemask):
    gdef = record["gdef"]
    # replication_number_feasibility_rounding(record, nodemask)
    strategy = { gdef.node[i].name: [int(ncclmask[gi])] + [ int(nodemask[gi, j]) for j in range(nodemask.shape[1]) ] for gi, group in enumerate(record["op_groups"]) for i in group }
    # info(strategy)
    leftout = [ gi for gi in range(len(record["op_groups"])) if np.sum(nodemask[gi, :]) == 0 ]
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]], sinks=["Adam"])
    tge.set_strategy(strategy)
    tge.fill_batchsize(record["batchsize"])
    tge.replace_placeholder(record["batchsize"])
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"])

    oom = [ i for i in range(len(mem)) if mem[i] > record["devices"][i][2] ]
    return np.sqrt(time / 1_000_000), oom, leftout

def sample_and_evaluate(record, placement_logit):
    placement_mask = sample(nodelogit)
    sqrt_time, oom, leftout = evaluate(record, placement_mask)

    if 'hist' not in record:
        record["hist"] = []

    if len(oom) == 0 and len(leftout) == 0:
        record["hist"].append(sqrt_time)
        record["hist"] = record["hist"][-100:]
        baseline = np.mean(record["hist"])
        advantage = -(sqrt_time - baseline) / baseline
    else:
        advantage = 0

    return ncclmask, nodemask, advantage, sqrt_time, oom, leftout

def f(arg):
    record, pheno = arg
    nodemask = pheno[:len(record['op_groups']) * len(record['devices'])]
    ncclmask = pheno[len(record['op_groups']) * len(record['devices']):]
    nodemask = np.reshape(nodemask, (len(record['op_groups']), len(record['devices'])))
    time, oom, leftout = evaluate(record, ncclmask, nodemask)
    nerror = len(oom) + len(leftout)
    return time * (1 + 10 * nerror)

def base_strategies(record):
    result = []

    ncgroups = len(record['op_groups'])
    ndevices = len(record['devices'])

    # 1: gpu0 + ps
    s = np.zeros((ncgroups, ndevices), dtype=np.int)
    for i in range(ncgroups):
        s[i, 0] = 1
    result.append((s, [0] * ncgroups))

    # 2: gpu0 + gpu1 + nccl
    s = np.zeros((ncgroups, ndevices), dtype=np.int)
    for i in range(ncgroups):
        s[i, 0] = 1
        s[i, 1] = 1
    result.append((s, [1] * ncgroups))

    # 3: gpu0 + gpu1 + gpu2 + gpu3 + nccl
    if ndevices >= 4:
        s = np.zeros((ncgroups, ndevices), dtype=np.int)
        for i in range(ncgroups):
            s[i, 0] = 1
            s[i, 1] = 1
            s[i, 2] = 1
            s[i, 3] = 1
        result.append((s, [1] * ncgroups))

    # 4. all + ps
    s = np.ones((ncgroups, ndevices), dtype=np.int)
    result.append((s, [0] * ncgroups))

    # 5. all + nccl
    s = np.ones((ncgroups, ndevices), dtype=np.int)
    result.append((s, [1] * ncgroups))

    # 6. first 4 * 2 + rest * 1 + nccl
    if ndevices >= 5:
        s = np.ones((ncgroups, ndevices), dtype=np.int)
        for i in range(ncgroups):
            s[i, 0] = 2
            s[i, 1] = 2
            s[i, 2] = 2
            s[i, 3] = 2
        result.append((s, [1] * ncgroups))

    return result

def replication_number_feasibility_rounding(record, nodemask):
    B = record["batchsize"]

    for i in range(nodemask.shape[0]):
        r = sum(nodemask[i, :])
        if B % r == 0:
            continue

        actions = []
        if r > 1 and B % (r - 1) == 0: # try remove one replica. Devices that have two replicas has double probability
            for j, k in enumerate(nodemask[i, :]):
                actions.extend([(j, -1)] * k)
        if B % (r + 1) == 0: # try add one replica, but only on devices that already have one
            for j, k in enumerate(nodemask[i, :]):
                if k == 1:
                    actions.append((j, 1))

        if len(actions) < 0: # heuristic failed, randomly remove replicas until feasible
            while B % sum(nodemask[i, :]) != 0:
                j = np.random.chioce(nodemask.shape[1])
                if nodemask[i, j] > 0:
                    nodemask[i, j] -= 1
            continue

        j, a = np.random.choice(actions)
        nodemask[i, j] += a

    return nodemask
