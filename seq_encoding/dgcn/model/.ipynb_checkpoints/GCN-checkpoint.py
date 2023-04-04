import torch
import torch.nn as nn
import dgl
# from torch_geometric.nn import RGCNConv, GraphConv
from dgl.nn.pytorch import RelGraphConv as RGCNConv
from dgl.nn.pytorch import GraphConv
# from .myRGCNConv import myRGCNConv
import dgcn
log = dgcn.utils.get_logger()

class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        # self.conv1 = myRGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    # def forward(self, node_features, edge_index, edge_norm, edge_type):

    #     x = self.conv1(node_features, edge_index, edge_type)
    #     x = self.conv2(x, edge_index, edge_weight=edge_norm)

    #     # x = self.conv1(node_features, edge_index, edge_type) 
    #     # log.info("x.shape = {}, {}".format(x.shape, edge_norm.view(-1, 1).shape) )
    #     # x = self.conv2(x, edge_index)


    #     return x


    # def forward(self,g, node_features, edge_index, edge_norm, edge_type):

    def forward(self,node_features, edge_index, edge_norm, edge_type):


        # log.info('norm samples = {}, {}, {}'.format(edge_norm[:10], edge_norm.shape, edge_index.shape))
        # create a DGLGraph heres
        # TODO 
        u = edge_index[0]
        v = edge_index[1]
        g = dgl.DGLGraph((u, v) )

        # x = self.conv1(g, node_features, edge_type)
        x = self.conv1(g, node_features, edge_type, norm=edge_norm.view(-1, 1))

        x = self.conv2(g, x, edge_weight=edge_norm)


        return x





















