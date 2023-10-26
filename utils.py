""" This file contains functions implemented for Random Entity Quantization """
import torch
from collections import Counter
import math


def compute_entropy(probabilities):
    result = 0.0
    for p in probabilities:
        if p == 0 or p == 1:
            result = 0 + result
        else:
            result = result + p * math.log2(p) + (1 - p) * math.log2(1 - p)
    return -result


def count_anchorset_entropy(anchor_selection_matrix):
    mask_int = anchor_selection_matrix.long()
    anchor_id_list = []
    for i in range(anchor_selection_matrix.shape[0]):
        anchor_id_list.append(torch.nonzero(mask_int[i]).squeeze(-1).tolist())
    set_list = [frozenset(anchor_id_list[i]) for i in range(0, len(anchor_id_list))]

    set_counter = Counter(set_list)

    probabilities_row = [count / len(set_list) for _, count in set_counter.items()]
    entropy_row = compute_entropy(probabilities_row)

    return entropy_row


def manhattan_distance(x, y):
    dist = torch.count_nonzero(x.unsqueeze(1) ^ y.unsqueeze(0), dim=2)
    return dist


def jaccard_distance(x, y):
    symmetric_difference = manhattan_distance(x, y)

    nonzero_x = torch.count_nonzero(x, dim=1)
    nonzero_y = torch.count_nonzero(y, dim=1)
    nonzero_both = nonzero_x.unsqueeze(1) + nonzero_y.unsqueeze(0)
    union = (nonzero_both + symmetric_difference) / 2 
    return symmetric_difference / union


def compute_anchorset_dist_topk(mask, gpu, topk=100):
    """ k-nn Jaccard Distance """
    dist_batch_size = 16
    mask = mask.to(gpu)
    tensor_batch = torch.split(mask, split_size_or_sections=dist_batch_size, dim=0)
    dist_list = []
    for batch in tensor_batch:
        # batch_dist = manhattan_distance(batch, mask) # [bs, num_ent]
        batch_dist = jaccard_distance(batch, mask)
        batch_dist, _ = torch.topk(batch_dist, k=topk, dim=-1, largest=False)
        avg_dist_batch = torch.div(torch.sum(batch_dist, dim=1), batch_dist.shape[1] - 1)
        dist_list.append(avg_dist_batch)
    dist = torch.cat(dist_list)
    avg_dist_all = torch.mean(dist)
    return avg_dist_all

