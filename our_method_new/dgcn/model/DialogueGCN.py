import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
from .functions import create_graph
from .RGT import RGTModel
import dgcn
import pandas as pd
import pickle
log = dgcn.utils.get_logger()


class DialogueGCN(nn.Module):

    def __init__(self, args):
        super(DialogueGCN, self).__init__()
#         self.vocab_size = -1
#         self.e_dim = 300
        u_dim = 100
        g_dim = 200
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6
        if ("Daily" in args.data) or ("MELD" in args.data):
            tag_size = 7
        cnn_filters=50
        cnn_kernel_sizes=(3, 4, 5)
        cnn_dropout=0.5
        ## GT layer 
        self.hidden_size = 80
        self.pos_enc_size = 2
        self.num_layer = 5
        self.num_headers = 8
        
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device     
        self.rnn = SeqContext(cnn_filters, cnn_kernel_sizes, cnn_dropout, u_dim, g_dim, args)
        
        self.edge_att = EdgeAtt(g_dim, args)
        self.gtm = RGTModel(tag_size, input_size = g_dim, hidden_size = self.hidden_size, args = args)
        self.clf = Classifier(g_dim + self.hidden_size, hc_dim, tag_size, args)
        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        textf, qmask, umask, label = [d.to(self.device) for d in data] if self.device != 'cpu' else data[:]
        # text len tensor = lengths
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        node_features = self.rnn(lengths, textf, umask) # [batch_size, mx_len, D_g]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, lengths, qmask, self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)
        
        graph_out = create_graph(features, edge_index, edge_norm, edge_type, self.device, self.pos_enc_size)
        graph_out = self.gtm(graph_out, graph_out.ndata['feat'], graph_out.ndata['PE']) # position encoding may be wrong!

        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        out = self.clf(torch.cat([features, graph_out], dim=-1), data[0])

        return out

    def get_loss(self, data):
        textf, qmask, umask, label = [d.cuda() for d in data] if self.device != 'cpu' else data[:]
        # text len tensor = lengths
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        graph_out, features = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 label, lengths)

        return loss
