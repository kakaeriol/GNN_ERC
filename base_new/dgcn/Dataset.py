import math
import random
import pandas as pd
import torch
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

# class Dataset:

#     def __init__(self, samples, batch_size,args):
#         self.samples = samples
#         self.batch_size = batch_size
#         self.num_batches = math.ceil(len(self.samples) / batch_size)
#         self.speaker_to_idx = {'M': 0, 'F': 1}
#         self.device = args.device
#     def __len__(self):
#         return self.num_batches

#     def __getitem__(self, index):
#         batch = self.raw_batch(index)
#         return self.padding(batch)

#     def raw_batch(self, index):
#         assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
#         batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

#         return batch

#     def padding(self, samples):
#         batch_size = len(samples)
#         text_len_tensor = torch.tensor([len(s.text) for s in samples]).long()
#         mx = torch.max(text_len_tensor).item()
#         text_tensor = torch.zeros((batch_size, mx, 100))
#         speaker_tensor = torch.zeros((batch_size, mx)).long()
#         labels = []
#         for i, s in enumerate(samples):
#             cur_len = len(s.text)
#             tmp = [torch.from_numpy(t).float() for t in s.text]
#             tmp = torch.stack(tmp)
#             text_tensor[i, :cur_len, :] = tmp
#             speaker_tensor[i, :cur_len] = torch.tensor([self.speaker_to_idx[c] for c in s.speaker])
#             labels.extend(s.label)

#         label_tensor = torch.tensor(labels).long()
#         data = {
#             "text_len_tensor": text_len_tensor,
#             "text_tensor": text_tensor,
#             "speaker_tensor": speaker_tensor,
#             "label_tensor": label_tensor
#         }

#         return data

#     def shuffle(self):
#         random.shuffle(self.samples)

import torch
from torch.utils.data import Dataset
import numpy as np
class my_DataSet(Dataset):
    """
    """
    def __init__(self, path , split, num_speaker=2):
        data = pd.read_pickle(path)
        data = data[split]
        self.feature = data["data_token"]
        self.speaker = data["speakers"]
        self.num_speaker = num_speaker
        self.labels  = data["emotions"]
        self.len = len(data["data_token"])
    def __getitem__(self, index):
        return torch.LongTensor(self.feature[index]), \
            torch.FloatTensor(self.speaker[index]), \
            torch.FloatTensor([1]*len(self.labels[index])), \
            torch.LongTensor(self.labels[index])
    
    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 1 else pad_sequence(dat[i], True) if i < 3 else torch.cat(dat[i].tolist(), -1) for i in
                dat]


