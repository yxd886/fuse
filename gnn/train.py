import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, sample_and_evaluate, base_strategies, f
from search import search
from utils import save, load, info

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

    for epoch in range(20000):
        record_id = np.random.randint(len(records))
        record = records[record_id]

        if 'reference' not in record:
            record['reference'] = []
            for nodemask, ncclmask in base_strategies(record):
                ncclmask = np.array(ncclmask)
                loss_env = f((record, np.hstack([np.reshape(nodemask, (-1, )), ncclmask])))
                record['reference'].append((loss_env, nodemask, ncclmask))
            save(records, "records")

        if 'elites' not in record:
            for loss_env, nodemask, ncclmask in record['reference']:
                if 'elites' not in record or record['elites'][0][0] > loss_env:
                    record['elites'] = [(loss_env, nodemask, ncclmask)]
            save(records, "records")

        op_feats     = tf.convert_to_tensor(record["op_feats"], dtype=tf.float32)
        device_feats = tf.convert_to_tensor(record["device_feats"], dtype=tf.float32)
        tensor_feats = tf.convert_to_tensor(record["tensor_feats"], dtype=tf.float32)
        link_feats   = tf.convert_to_tensor(record["link_feats"], dtype=tf.float32)
        place_feats  = tf.convert_to_tensor(record["place_feats"], dtype=tf.float32)
        model.set_graph(record["graph"])

        # search
        if epoch > 100 and epoch % 50 == 0:
            nodelogit, nccllogit = model([op_feats, device_feats, tensor_feats, link_feats, place_feats], training=False)
            nodep = tf.nn.softmax(nodelogit).numpy()
            ncclp = tf.math.sigmoid(nccllogit).numpy()
            loss_env, nodemask, ncclmask = search(record, nodep, ncclp)
            if loss_env < record['elites'][-1][0] * 1.05:
                record['elites'].append((loss_env, nodemask, ncclmask))
                record["elites"] = record["elites"][-4:]

            info(record_id, loss_env, [ x for x, _, _ in record['reference'] ])

        # learn
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            nodelogit, nccllogit = model([op_feats, device_feats, tensor_feats, link_feats, place_feats], training=True)

            # info(tf.nn.softmax(nodelogit).numpy())
            # info(nodelogit.numpy())

            loss = 0
            for loss_env, nodemask, ncclmask in record['elites']:
                nodemask = tf.one_hot(nodemask, depth=3).numpy()
                loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(nodemask.astype(np.float32), nodelogit))
                loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ncclmask.astype(np.float32), nccllogit))
            loss /= len(record['elites'])

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            info(record_id, loss.numpy())

            grads = tape.gradient(loss, model.trainable_weights)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # checkpoint
        if epoch % 50 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")
