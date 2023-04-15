import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
from .TransformerEncoder import DialogueTransformer
import numpy as np, itertools, random, copy, math
import pickle

class CNNFeatureExtractor(nn.Module):
    """
    Module from DialogueRNN
    """
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout, args):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(args.device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size
        self.device = args.device

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float().to(self.device))
        self.embedding.weight.requires_grad = False

    def forward(self, x, umask):
        num_utt, batch, num_words = x.size()
        x = x.type(torch.LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words)  # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        x = x.to(self.device)
        emb = self.embedding(x)  # (num_utt * batch, num_words) -> (num_utt * batch, num_words, 300)
        emb = emb.transpose(-2,
                            -1).contiguous()  # (num_utt * batch, num_words, 300)  -> (num_utt * batch, 300, num_words)

        convoluted = [F.relu(conv(emb)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted]
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated)))  # (num_utt * batch, 150) -> (num_utt * batch, 100)
        features = features.view(num_utt, batch, -1)  # (num_utt * batch, 100) -> (num_utt, batch, 100)
        mask = umask.unsqueeze(-1).type(torch.FloatTensor)  # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1)  # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim)  # (num_utt, batch, 1) -> (num_utt, batch, 100)
        mask = mask.to(self.device)
        features = (features * mask)  # (num_utt, batch, 100) -> (num_utt, batch, 100)
        features = features.permute(1, 0, 2)

        return features
class SeqContext(nn.Module):

    def __init__(self, cnn_filters, cnn_kernel_sizes, cnn_dropout, u_dim, g_dim, args):
        super(SeqContext, self).__init__()
        
        self.input_size = u_dim
        self.hidden_dim = g_dim
        self.model_type = args.rnn
        
            
        glv_pretrained = pickle.load(open(args.pretrained_word_vectors, 'rb'))
        vocab_size, e_dim = glv_pretrained['embedding'].shape

        # output size of cnn layer is the input size of rnn
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, e_dim, u_dim, cnn_filters, cnn_kernel_sizes, cnn_dropout, args)
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(glv_pretrained['embedding'])  
        if args.rnn in ['lstm', 'gru']:
            if args.rnn == "lstm":
                self.rnn = nn.LSTM(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                                   bidirectional=True, num_layers=2, batch_first=True)
            elif args.rnn == "gru":
                self.rnn = nn.GRU(self.input_size, self.hidden_dim // 2, dropout=args.drop_rate,
                                  bidirectional=True, num_layers=2, batch_first=True)
        elif args.rnn == 'transformer':
            self.transformer_encoder = DialogueTransformer(self.input_size, self.hidden_dim, 1, 4, self.hidden_dim, args.drop_rate)


                
    def forward(self, text_len_tensor, textft, umask):
        
        if self.model_type in ("lstm", "gru"):
            text_tensor = self.cnn_feat_extractor(textft, umask)
            packed = pack_padded_sequence(
                text_tensor,
                text_len_tensor,
                batch_first=True,
                enforce_sorted=False
            )
            try:
                rnn_out, (_, _) = self.rnn(packed, None)
            except:
                rnn_out, _ = self.rnn(packed, None)
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
            
        elif self.model_type == "transformer":
            text_tensor = self.cnn_feat_extractor(textft, umask)
            # text_tensor should have shape (n, S, E) where n is batch size, S is sequence length, and E is the embedding size
            transformer_out = self.transformer_encoder(text_tensor)
            return transformer_out

            return rnn_out
