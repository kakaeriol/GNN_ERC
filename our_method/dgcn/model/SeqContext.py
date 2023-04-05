import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .TransformerEncoder import DialogueTransformer

class SeqContext(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        self.model_type = args.rnn
        if args.rnn == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == 'transformer':
            self.rnn = DialogueTransformer(self.input_size, self.hidden_dim, 1, 4, self.hidden_dim, args.drop_rate)

    def forward(self, text_len_tensor, text_tensor):
        if self.model_type == "transformer":
            transformer_out = self.rnn(text_tensor)
            return transformer_out
        else:
        
            packed = pack_padded_sequence(
                text_tensor,
                text_len_tensor.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

            rnn_out, _ = self.rnn(packed, None)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

            return rnn_out