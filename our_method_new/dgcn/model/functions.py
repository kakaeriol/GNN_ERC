import numpy as np
import torch

import dgcn

log = dgcn.utils.get_logger()
import dgl



def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j], wp, wf))
    
    edge_weights = att_model(features, lengths, edge_ind)
    for j in range(batch_size):
        cur_len = lengths[j]
        for icur in range(cur_len):
            node_features.append(features[j, icur, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])
            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item() #
            # speaker0 = (qmask[item1[0], j, :] == 1).nonzero()[0][0].tolist()
            # speaker1 = (qmask[item1[1], j, :] == 1).nonzero()[0][0].tolist()
            
            if item[0] < item[1]:
                c = '0'
            else:
                c = '1'
            edge_type.append(edge_type_to_idx[str(int(speaker1) )+ str(int(speaker2)) + c])


    
    # node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    node_features = torch.stack(node_features).to(device)
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)

def create_graph(node_features, edge_index, edge_norm, edge_type, device, pos_enc_size=2):
    """
    Return the graph form from graph features
    """
    graph_out = dgl.graph((edge_index[0], edge_index[1]))
    graph_out.norm = edge_norm
    graph_out.ndata['feat'] = node_features
    graph_out.ndata['PE'] = dgl.laplacian_pe(graph_out, k=pos_enc_size, padding=True).to(device)
    graph_out.edata['feat'] = torch.ones_like(edge_index[0])
    graph_out.edata['norm'] = edge_norm
    graph_out.edata['rel_type'] = edge_type
    # graph_out.edge_index_lengths = edge_index_lengths
    return graph_out