
import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info, positional_encoding


class SingleGConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, out_feats, activation=None):
        super(SingleGConv, self).__init__()
        self.activation = activation
        self.all_etypes = ["prev", "succ","to_final_edge_feats"]

        self.layers = { etype: tf.keras.layers.Dense(out_feats, activation=None) for etype in self.all_etypes }

    def call(self, graph, instruction_feats, final_feats,edge_feats):
        instruction_dst,computation_dst,final_dst = [], [], []
        for stype, etype, dtype in graph.canonical_etypes:
            g = graph[etype].local_var()

            if stype == 'instruction':
                g.srcdata['i'] = instruction_feats
            elif stype== "final":
                g.srcdata['i'] = final_feats


            g.apply_edges(fn.copy_u('i', 's'))
            edata = tf.concat([g.edata.pop('s'), edge_feats[etype]], axis=1)
            g.edata['e'] = self.layers[etype](edata)
            g.update_all(fn.copy_e('e', 'm'), fn.mean(msg='m', out='o'))

            if dtype == 'instruction':
                instruction_dst.append(g.dstdata['o'])
            elif dtype == "final":
                final_dst.append(g.dstdata['o'])
        instruction_dst = tf.math.add_n(instruction_dst) / len(instruction_dst)
        final_dst = tf.math.add_n(final_dst) / len(final_dst)
        return self.activation(instruction_feats + instruction_dst), self.activation(final_feats+final_dst)

class SingleModel(tf.keras.Model):
    def __init__(self):
        super(SingleModel, self).__init__()

        node_hidden = 64
        edge_hidden = 8
        self.all_etypes = [ "prev", "succ","to_final_edge_feats"]


        self.instruction_trans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.elu)
        self.final_trans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.elu)
        self.edge_trans = { etype: tf.keras.layers.Dense(edge_hidden, activation=tf.nn.elu) for etype in self.all_etypes }

        self.gconv_layers = [
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu),
            SingleGConv(node_hidden, tf.nn.elu)
        ]

        self.final_ranks = [
            tf.keras.layers.Dense(node_hidden, activation=tf.nn.elu),
            tf.keras.layers.Dense(node_hidden, activation=tf.nn.elu),
            tf.keras.layers.Dense(1, activation=None),

        ]


    def set_graph(self, graph):
        self.graph = graph.to('gpu:0')
        #self.graph = graph

    def call(self, inputs):
        [instruction_feats,final_feats, instruction_edge_feats ,to_final_edge_feats] = inputs

        instruction_feats = self.instruction_trans(instruction_feats)
        final_feats = self.final_trans(final_feats)


        edge_feats = {
            "prev": instruction_edge_feats,
            "succ": instruction_edge_feats,
            "to_final_edge_feats":to_final_edge_feats
        }
        edge_feats = { etype: self.edge_trans[etype](edge_feats[etype]) for etype in self.all_etypes  }

        for gconv_layer in self.gconv_layers:
            instruction_feats,final_feats = gconv_layer(self.graph, instruction_feats, final_feats,edge_feats)

        for final_rank in self.final_ranks:
            final_feats  = final_rank(final_feats)

        return  tf.squeeze(final_feats, axis=1)