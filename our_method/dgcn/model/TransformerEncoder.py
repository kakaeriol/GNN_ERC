import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DialogueTransformer(nn.Module):
    def __init__(self, input_size, embedding_size, num_layers, num_heads, hidden_size, dropout):
        super().__init__()
        self.embedding_size = embedding_size
        
        self.embedding_layer1 = nn.Linear(input_size, embedding_size)
        # self.embedding_layer2 = nn.Linear(input_size, embedding_size)
        
        self.positional_encoding = PositionalEncoding(embedding_size, dropout)
        
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout) for _ in range(num_layers)])
        
    def forward(self, inputs, umask):
        inp = self.embedding_layer1(inputs)
        x = self.positional_encoding(inp)
        
        for layer in self.transformer_layers:
            x = layer(x, umask)
        
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_len=250):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout,batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
            nn.Dropout(p=dropout)
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, umask):
        residual = x
        x = self.norm1(x)
        key_padding_mask = umask != 1 # (BS * num_utt)
        att_mask = torch.tril(torch.ones((umask.shape[1], umask.shape[1]))).to(x.device) # num_utt * num_utt
        x, _ = self.self_attn(x, x, x,key_padding_mask=key_padding_mask)
        x, _ = self.self_attn(x, x, x,key_padding_mask=key_padding_mask, attn_mask=att_mask)
        x = F.softmax(x, dim=-1)
        
        # multiply with a 
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
    
        return x