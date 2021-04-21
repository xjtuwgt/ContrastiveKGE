#####Entity/relation embedding#####
import torch
from torch import Tensor
from torch import nn
import torch.nn.init as INIT
from dgl.nn.pytorch.utils import Identity

class ExternalEmbedding(nn.Module):
    def __init__(self, num: int, dim: int, project_dim: int = None):
        super(ExternalEmbedding, self).__init__()
        self.num = num
        self.dim = dim
        self.proj_dim = project_dim
        self.emb = nn.Parameter(torch.zeros(num, self.dim), requires_grad=True)
        if self.proj_dim is not None and self.proj_dim > 0:
            self.projection = torch.nn.Linear(self.dim, self.proj_dim, bias=False)
        else:
            self.projection = Identity()

    def init(self, emb_init):
        """Initializing the embeddings.
        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        INIT.uniform_(self.emb, -emb_init, emb_init)
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_normal_(self.projection.weight, gain=gain)

    def _embed(self, embeddings):
        embeddings = self.projection(embeddings)
        return embeddings

    def embed(self, indexes):
        return self._embed(self.emb[indexes])

    def embed_all(self):
        return self._embed(self.emb)

    def forward(self, idx: Tensor):
        data = self.embed(idx)
        return data

if __name__ == '__main__':
    entitiy_emb = ExternalEmbedding(num=5, dim=6, project_dim=2)
    entitiy_emb.init(emb_init=0.1)
    idx = torch.LongTensor([[1,2,3],[1,2,3]])
    print(entitiy_emb.emb.data.shape)
    print(entitiy_emb.emb.norm(2))
    # print(entitiy_emb.emb)

    # print(entitiy_emb.state_sum)
    print(entitiy_emb(idx))

