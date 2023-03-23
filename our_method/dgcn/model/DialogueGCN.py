import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GraphTransform import GTModel
from .graph_functions import batch_graphify

import dgcn
log = dgcn.utils.get_logger()


class DialogueGCN(nn.Module):
    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        u_dim = 100
        g_dim = 200
        tag_size = 6
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
        self.rnn = SeqContext(u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gtm = GTModel(tag_size, input_size= g_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        node_features = self.rnn(data["text_len_tensor"], data["text_tensor"]) # [batch_size, mx_len, D_g]
        graph_out = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device, self.pos_enc_size)
        return graph_out

    def forward(self, data):
        graph_out= self.get_rep(data)
        out = self.gtm(graph_out, graph_out.ndata['feat'], graph_out.ndata['PE'])
        # out = torch.argmax(out, dim=-1)
        return self.softmax(out)

#     def loss(self, y_scores, y_labels):
#         loss = nn.CrossEntropyLoss()(y_scores, y_labels)
#         return loss        
        
#     def accuracy(self, scores, targets):
#         scores = scores.detach().argmax(dim=1)
#         acc = (scores==targets).float().sum().item()
#         return acc
    
#     def update(self, lr):       
#         update = torch.optim.Adam( self.parameters(), lr=lr )
#         return update
