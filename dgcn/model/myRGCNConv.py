import torch
import torch.nn as nn
import dgl


class myGraph(dgl.DGLGraph):
    def __init__(self, node_features, edge_index, edge_norm, edge_type):
        u = edge_index[0]
        v = edge_index[1]
        super(myGraph, self).__init__((u, v), num_nodes=node_features.shape[0])
        self.etypes = edge_type
        self.edata['norm'] = edge_norm
        self.ndata['h'] = node_features
