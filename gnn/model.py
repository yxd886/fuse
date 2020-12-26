import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info, positional_encoding

all_etypes = ["link", "prev", "succ", "place", "serve"]

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, out_feats, activation=None):
        super(GConv, self).__init__()
        self.activation = activation
        self.layers = { etype: tf.keras.layers.Dense(out_feats, activation=None) for etype in all_etypes }

    def call(self, graph, op_feats, device_feats, edge_feats):
        op_dst, device_dst = [], []
        for stype, etype, dtype in graph.canonical_etypes:
            g = graph[etype].local_var()

            if stype == 'op':
                g.srcdata['i'] = op_feats
            elif stype == 'device':
                g.srcdata['i'] = device_feats

            g.apply_edges(fn.copy_u('i', 's'))
            edata = tf.concat([g.edata.pop('s'), edge_feats[etype]], axis=1)
            g.edata['e'] = self.layers[etype](edata)
            g.update_all(fn.copy_e('e', 'm'), fn.mean(msg='m', out='o'))

            if dtype == 'op':
                op_dst.append(g.dstdata['o'])
            elif dtype == 'device':
                device_dst.append(g.dstdata['o'])

        op_dst = tf.math.add_n(op_dst) / len(op_dst)
        device_dst = tf.math.add_n(device_dst) / len(device_dst)

        return self.activation(op_feats + op_dst), self.activation(device_feats + device_dst)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        node_hidden = 64
        edge_hidden = 8

        self.op_trans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.elu)
        self.device_trans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.elu)
        self.edge_trans = { etype: tf.keras.layers.Dense(edge_hidden, activation=tf.nn.elu) for etype in all_etypes }

        self.gconv_layers = [
            GConv(node_hidden, tf.nn.elu),
            GConv(node_hidden, tf.nn.elu),
            GConv(node_hidden, tf.nn.elu),
            GConv(node_hidden, tf.nn.elu),
            GConv(node_hidden, tf.nn.elu),
            GConv(node_hidden, tf.identity)
        ]

        self.final_place = tf.keras.layers.Dense(3, activation=None)
        self.final_nccl = tf.keras.layers.Dense(1, activation=None)

    def set_graph(self, graph):
        # self.graph = graph.to('gpu:0')
        self.graph = graph

    def call(self, inputs):
        [op_feats, device_feats, tensor_feats, link_feats, place_feats] = inputs

        op_feats = self.op_trans(op_feats)
        device_feats = self.device_trans(device_feats)

        edge_feats = {
            "link": link_feats,
            "prev": tensor_feats,
            "succ": tensor_feats,
            "place": place_feats,
            "serve": place_feats
        }
        edge_feats = { etype: self.edge_trans[etype](edge_feats[etype]) for etype in all_etypes }

        for gconv_layer in self.gconv_layers:
            op_feats, device_feats = gconv_layer(self.graph, op_feats, device_feats, edge_feats)

        g = self.graph['place'].local_var()
        g.srcdata['i'] = op_feats
        g.dstdata['i'] = device_feats
        g.edata['i'] = edge_feats['place']
        def decision(edge):
            return { 'd': self.final_place(tf.concat([edge.src['i'], edge.data['i'], edge.dst['i']], axis=1))  }
        g.apply_edges(decision)

        return g.edata['d'], tf.squeeze(self.final_nccl(op_feats), axis=1)
