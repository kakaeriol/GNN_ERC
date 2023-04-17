import math
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

import torch
import numpy as np
class my_DataSet(Dataset):
    """
    """
    def __init__(self, path , split, num_speaker=2):
        data = pd.read_pickle(path)
        # pickle = load(open(path, 'rb'))
        data = data[split]
        self.feature = data["data_token"]
        # self.text_len = [len(i) for i in self.feature]
        self.speaker = data["speakers"]
        self.num_speaker = num_speaker
        self.labels  = data["emotions"]
        self.len = len(data["data_token"])
    def __getitem__(self, index):
        # return: text_tensor, speaker_tensor, conv_len_mask, label_tensor
        return torch.LongTensor(self.feature[index]), \
            torch.FloatTensor(self.speaker[index]), \
            torch.FloatTensor([1]*len(self.labels[index])), \
            torch.LongTensor(self.labels[index])
            # torch.LongTensor(self.text_len[index])
    
    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 1 else pad_sequence(dat[i], True) if i < 3 else torch.cat(dat[i].tolist(), -1) for i in
                dat]




