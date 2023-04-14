import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
from .CNN import CNNFeatureExtractor
import dgcn
import pandas as pd
log = dgcn.utils.get_logger()


class DialogueGCN(nn.Module):

    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        voca_size = 250
        e_dim = 300
        u_dim = 100
        g_dim = 200
        h1_dim = 100
        h2_dim = 100
        hc_dim = 100
        tag_size = 6
        
        cnn_filters=50
        cnn_kernel_sizes=(3, 4, 5)
        cnn_dropout=0.5

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        self.cnn_feat_extractor = CNNFeatureExtractor(voca_size, e_dim, u_dim, cnn_filters,cnn_kernel_sizes,cnn_dropout, args)
        self.rnn = SeqContext(u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gcn = GCN(g_dim, h1_dim, h2_dim, args)
        self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)
        glv_pretrained = pd.read_pickle(args.pretrained_word_vectors)
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(glv_pretrained['embedding'])
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
        # qmask = qmask.permute(1, 0) # change to #batch x L 
        cnn_node_fts = self.cnn_feat_extractor(textf, umask)
        node_features = self.rnn(lengths, cnn_node_fts) # [batch_size, mx_len, D_g]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, lengths, qmask, self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

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
