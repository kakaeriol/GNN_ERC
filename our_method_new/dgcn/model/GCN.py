import torch.nn as nn
import dgl
from dgl.nn.pytorch import RelGraphConv as RGCNConv
from dgl.nn.pytorch import GraphConv
# from torch_geometric.nn import GraphConv
# from .RGCN import RGCNConv

class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)
        if args.device != 'cpu':
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        g = dgl.graph((edge_index[0], edge_index[1]))
        g.norm = edge_norm
        x = self.conv1(g, node_features, edge_type)
        x = self.conv2(g, x)

        return x






















