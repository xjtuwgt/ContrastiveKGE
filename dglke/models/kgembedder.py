#####Entity/relation embedding#####

import os
import torch
import numpy as np
from torch import Tensor
from torch import nn
import torch.nn.init as INIT

class ExternalEmbedding(nn.Module):
    def __init__(self, num: int, dim: int):
        super(ExternalEmbedding, self).__init__()
        self.num = num
        self.dim = dim
        self.emb = nn.Embedding(num_embeddings=num, embedding_dim=dim)


    def init(self, emb_init):
        """Initializing the embeddings.
        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        INIT.uniform_(self.emb.weight.data, -emb_init, emb_init)

    def forward(self, idx: Tensor):
        data = self.emb(idx)
        return data

    def save(self, path, name):
        """Save embeddings.
        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name + '.npy')
        np.save(file_name, self.emb.weight.data.cpu().detach().numpy())

    def load(self, path, name):
        """Load embeddings.
        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name + '.npy')
        self.emb.weight.data.copy_(torch.from_numpy(np.load(file_name)))


if __name__ == '__main__':
    entitiy_emb = ExternalEmbedding(num=5, dim=6)
    entitiy_emb.init(emb_init=0.1)
    idx = torch.LongTensor([[1,2,3],[1,2,3]])
    print(entitiy_emb.emb.weight.data.shape)
    # print(entitiy_emb.emb)

    # print(entitiy_emb.state_sum)
    print(entitiy_emb(idx))

