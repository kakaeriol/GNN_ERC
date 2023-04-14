import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np, itertools, random, copy, math

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