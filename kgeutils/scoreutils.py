import torch
from torch import Tensor as T
import torch.nn as nn

def batched_l2_dist(a: T, b: T):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = torch.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res

def batched_l1_dist(a: T, b: T):
    res = torch.cdist(a, b, p=1)
    return res

class TransEScore(nn.Module):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """
    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.dist_ord = 1
        else:  # default use l2
            self.dist_ord = 2

    def score(self, head, tail, relation, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        return {'score': self.gamma - torch.norm(score, p=self.dist_ord, dim=-1)}

class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self):
        super(DistMultScore, self).__init__()

    def score(self, head, tail, relation, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': torch.sum(score, dim=-1)}

class SimplEScore(nn.Module):
    """SimplE score function
    Paper link: http://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """
    def __init__(self):
        super(SimplEScore, self).__init__()

    def score(self, head, tail, relation, mode):
        head_i, head_j = torch.chunk(head, 2, dim=-1)
        tail_i, tail_j = torch.chunk(tail, 2, dim=-1)
        rel, rel_inv = torch.chunk(relation, 2, dim=-1)
        if mode == 'head-batch':
            forward_score = head_i * (rel * tail_j)
            backward_score = tail_i * (rel_inv * head_j)
        else:
            forward_score = (head_i * rel) * tail_j
            backward_score = (tail_i * rel_inv) * head_j
        # clamp as official implementation does to avoid NaN output
        # might because of gradient explode
        score = torch.clamp(1 / 2 * (forward_score + backward_score).sum(-1), -20, 20)
        return {'score': score}