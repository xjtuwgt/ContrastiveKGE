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
        self.emb = torch.empty(num, dim, dtype=torch.float32)
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()

    def init(self, emb_init):
        """Initializing the embeddings.
        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        INIT.uniform_(self.emb, -emb_init, emb_init)
        INIT.zeros_(self.state_sum)

    def forward(self, idx: Tensor):
        data = self.emb[idx]
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
        np.save(file_name, self.emb.cpu().detach().numpy())

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
        self.emb = torch.Tensor(np.load(file_name))


class InferEmbedding:
    def __init__(self, device):
        self.device = device

    def load(self, path, name):
        """Load embeddings.
        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name+'.npy')
        self.emb = torch.Tensor(np.load(file_name))

    def load_emb(self, emb_array):
        """Load embeddings from numpy array.
        Parameters
        ----------
        emb_array : numpy.array  or torch.tensor
            Embedding array in numpy array or torch.tensor
        """
        if isinstance(emb_array, np.ndarray):
            self.emb = torch.Tensor(emb_array)
        else:
            self.emb = emb_array

    def __call__(self, idx):
        return self.emb[idx].to(self.device)

if __name__ == '__main__':
    entitiy_emb = ExternalEmbedding(num=5, dim=6)
    entitiy_emb.init(emb_init=0.1)
    idx = torch.LongTensor([[1,2,3],[1,2,3]])
    print(entitiy_emb.emb.shape)
    print(entitiy_emb.emb)

    print(entitiy_emb.state_sum)
    print(entitiy_emb(idx))

