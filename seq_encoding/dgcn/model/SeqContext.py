import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .TransformerEncoder import DialogueTransformer

class SeqContext(nn.Module):

    def __init__(self, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        self.input_size = u_dim
        self.hidden_dim = g_dim
        self.model_type = args.model_type
        if args.model_type == "lstm":
            self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                               bidirectional=True, num_layers=2, batch_first=True)
        elif args.model_type == "gru":
            self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                              bidirectional=True, num_layers=2, batch_first=True)
            
            
        elif self.model_type == 'transformer':
            # self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=4, dim_feedforward=self.hidden_dim, dropout=args.drop_rate, activation='relu')
            # self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
            self.transformer_encoder = DialogueTransformer(self.input_size, self.hidden_dim, 1, 4, self.hidden_dim, args.drop_rate)

    def forward(self, text_len_tensor, text_tensor):
        if self.model_type in ("lstm", "gru"):
            packed = pack_padded_sequence(
                text_tensor,
                text_len_tensor.to('cpu'),
                batch_first=True,
                enforce_sorted=False
            )
            rnn_out, _ = self.rnn(packed, None)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
            return rnn_out
        elif self.model_type == "transformer":
            # text_tensor should have shape (n, S, E) where n is batch size, S is sequence length, and E is the embedding size
            transformer_out = self.transformer_encoder(text_tensor)
            return transformer_out
