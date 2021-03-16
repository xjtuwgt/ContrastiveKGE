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
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else:  # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - torch.norm(score, p=self.dist_ord, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb + rel_emb).unsqueeze(2) - tail_emb.unsqueeze(0).unsqueeze(0)
        return self.gamma - torch.norm(score, p=self.dist_ord, dim=-1)

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, chunk_size, hidden_dim)
                return gamma - self.neg_dist_func(tails, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                return gamma - self.neg_dist_func(heads, tails)
            return fn

class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': torch.sum(score, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb * rel_emb).unsqueeze(2) * tail_emb.unsqueeze(0).unsqueeze(0)

        return torch.sum(score, dim=-1)

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = torch.transpose(heads, 1, 2)
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return torch.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = torch.transpose(tails, 1, 2)
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return torch.bmm(tmp, tails)
            return fn


class SimplEScore(nn.Module):
    """SimplE score function
    Paper link: http://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """
    def __init__(self):
        super(SimplEScore, self).__init__()

    def edge_func(self, edges):
        head_i, head_j = torch.chunk(edges.src['emb'], 2, dim=-1)
        tail_i, tail_j = torch.chunk(edges.dst['emb'], 2, dim=-1)
        rel, rel_inv = torch.chunk(edges.data['emb'], 2, dim=-1)
        forward_score = head_i * rel * tail_j
        backward_score = tail_i * rel_inv * head_j
        # clamp as official implementation does to avoid NaN output
        # might because of gradient explode
        score = torch.clamp(1 / 2 * (forward_score + backward_score).sum(-1), -20, 20)
        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_i, head_j = torch.chunk(head_emb.unsqueeze(1), 2, dim=-1)
        tail_i, tail_j = torch.chunk(tail_emb.unsqueeze(0).unsqueeze(0), 2, dim=-1)
        rel, rel_inv = torch.chunk(rel_emb.unsqueeze(0), 2, dim=-1)
        forward_tmp = (head_i * rel).unsqueeze(2) * tail_j
        backward_tmp = (head_j * rel_inv).unsqueeze(2) * tail_i
        score = (forward_tmp + backward_tmp) * 1 / 2
        return torch.sum(score, dim=-1)

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tail_i = tails[..., :hidden_dim // 2]
                tail_j = tails[..., hidden_dim // 2:]
                rel = relations[..., : hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (rel * tail_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * tail_i).reshape(num_chunks, chunk_size, hidden_dim//2)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = torch.transpose(heads, 1, 2)
                head_i = heads[..., :hidden_dim // 2, :]
                head_j = heads[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (torch.bmm(forward_tmp, head_i) + torch.bmm(backward_tmp, head_j))
                score = torch.clamp(tmp, -20, 20)
                return score
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                head_i = heads[..., :hidden_dim // 2]
                head_j = heads[..., hidden_dim // 2:]
                rel = relations[..., :hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (head_i * rel).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * head_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = torch.transpose(tails, 1, 2)
                tail_i = tails[..., :hidden_dim // 2, :]
                tail_j = tails[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (torch.bmm(forward_tmp, tail_j) + torch.bmm(backward_tmp, tail_i))
                score = torch.clamp(tmp, -20, 20)
                return score
            return fn