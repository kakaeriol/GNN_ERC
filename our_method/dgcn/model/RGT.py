import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm

class MLP_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim, L=2): # L = nb of hidden layers
        super(MLP_layer, self).__init__()
        list_FC_layers = [ nn.Linear( input_dim, input_dim, bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim, output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))
    
class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)
    
class RGTModel_Final_layer(nn.Module):
    """
    Graph transform model - Need to add relational here
    """
    def __init__(
        self,
        out_size, # 6
        input_size=200, # g_dim
        hidden_size=80,
        pos_enc_size=2,
        num_layers=8,
        num_heads=8,
    ):
        super().__init__()
        self.embedding_h =  nn.Linear(input_size, hidden_size)#dgl.nn.GATConv(input_dim, hidden_size, num_heads=num_heads)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.predictor = MLP_layer(hidden_size, out_size)

    def forward(self, g, X, pos_enc): ## Need to change to message function later!!
        """  
        """
        N = g.num_nodes()
        h = self.embedding_h(X) + self.pos_linear(pos_enc)
        ll = []
        h_out = torch.zeros_like(h)
        edges = g.edges()
        edges_norm = g.edata['norm']
        for itype in g.edata['rel_type'].unique():
            src = edges[0][g.edata['rel_type'] == itype]
            des = edges[1][g.edata['rel_type'] == itype]
            val = edges_norm[g.edata['rel_type'] == itype]
            # ---
            indices = torch.stack((src, des))
            A_k =  dglsp.spmatrix(indices, shape=(N, N))
            # A_k =  dglsp.spmatrix(indices=indices, val =val, shape=(N, N))
            hk = h #torch.clone(h)
            for layer in self.layers:
                hk = layer(A_k, hk)
            ll.append(hk) # <-- should we try the voting for each A instead of multiply!!!
            h_out = h_out = self.batchnorm1(hk + h_out) ## instead of this one adding the concat layer
        return self.predictor(h_out)
    
### RELATIONAL IN GTLAYER AT MULTIHEAD LAYER
class RGTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, g, h):
        N = g.num_nodes()
        h_out = torch.zeros_like(h)
        ll = []
        edges = g.edges()
        edges_norm = g.edata['norm']
        for itype in g.edata['rel_type'].unique():
            src = edges[0][g.edata['rel_type'] == itype]
            des = edges[1][g.edata['rel_type'] == itype]
            val = edges_norm[g.edata['rel_type'] == itype]
            # ---
            indices = torch.stack((src, des))
            A_k =  dglsp.spmatrix(indices, shape=(N, N))
            hk = h #torch.clone(h)
            hk = self.MHA(A_k, hk)
            ll.append(hk)
            h_out = self.batchnorm1(hk + h_out) # need to change to linear layer later!
        h = self.batchnorm1(h + h_out)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h
        return self.batchnorm2(h)
    
class RGTModel(nn.Module):
    """
    Graph transform model - Need to add relational here
    """
    def __init__(
        self,
        out_size, # 6
        input_size=200, # g_dim
        hidden_size=80,
        pos_enc_size=2,
        num_layers=8,
        num_heads=8,
    ):
        super().__init__()
        self.embedding_h =  nn.Linear(input_size, hidden_size)#dgl.nn.GATConv(input_dim, hidden_size, num_heads=num_heads)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [RGTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.predictor = MLP_layer(hidden_size, out_size)

    def forward(self, g, X, pos_enc):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        h = self.embedding_h(X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(g, h)
        return self.predictor(h)