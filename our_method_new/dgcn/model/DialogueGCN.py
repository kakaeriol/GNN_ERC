import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GraphTransform import GTModel
from .graph_functions import batch_graphify
from .graph_functions import create_graph
from .RGT import RGTModel_Final_layer
from .RGT import RGTModel
import dgcn
log = dgcn.utils.get_logger()


class DialogueGCN(nn.Module):
    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        u_dim = 100
        g_dim = 200
        tag_size = 6
        if ("Daily" in args.data) or ("MELD" in args.data):
            tag_size = 7
        # ---
        cnn_filters=50
        cnn_kernel_sizes=(3, 4, 5)
        cnn_dropout=0.5

        # GTLayer
        self.hidden_size = 80
        self.pos_enc_size = 2
        self.num_layer = 8
        self.num_headers = 8
        #
        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device
        
        #
        self.rnn = SeqContext(cnn_filters, cnn_kernel_sizes, cnn_dropout, u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        # self.gtm = GTModel(tag_size, input_size= g_dim) ## adding if else here later
        if args.Rtype == 'Final':
            self.gtm = RGTModel_Final_layer(tag_size, input_size= g_dim, args=args) #v00
        elif args.Rtype == "MHA":
            self.gtm = RGTModel(tag_size, input_size= g_dim, args=args) #v01
        self.softmax = nn.LogSoftmax(dim=1)
        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        textf, qmask, umask, label = [d.to(self.device) for d in data] if self.device != 'cpu' else data[:]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        node_features = self.rnn(lengths, textf, umask) # [batch_size, mx_len, D_g]
        node_features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, lengths, qmask, self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)
        graph_out = create_graph(node_features, edge_index, edge_norm, edge_type, self.device, self.pos_enc_size)
        return graph_out

    def forward(self, data):
        graph_out= self.get_rep(data)
        textf, qmask, umask, label = [d.to(self.device) for d in data] if self.device != 'cpu' else data[:]
        graph_out.data_length = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        # graph_out.data_length = data[-1] # use in case of MAH in last layer
        out = self.gtm(graph_out, graph_out.ndata['feat'], graph_out.ndata['PE']) # Graph Transform
        # out = torch.argmax(out, dim=-1)
        return self.softmax(out)

