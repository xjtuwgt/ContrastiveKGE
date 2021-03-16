import os
import dgl
import torch
from graphutils.partition_utils import get_partition_list
import numpy as np

class ClusterIter(object):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, dn, g, psize, batch_size):
        """Initialize the sampler.
        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        """
        self.psize = psize
        self.batch_size = batch_size
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join('./datasets/', dn + '_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs('./datasets/', exist_ok=True)
                self.par_li = get_partition_list(g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(g, psize)
        par_list = []
        total_nodes = 0
        for p in self.par_li:
            total_nodes = total_nodes + len(p)
            par = torch.Tensor(p)
            par_list.append(par)
        self.par_list = par_list
        print('Partition number = {} over {} nodes on graph with {} nodes'.format(len(par_list), total_nodes, g.num_nodes()))

    def __len__(self):
        return self.psize

    def __getitem__(self, idx):
        return self.par_li[idx]

def subgraph_collate_fn(g, batch):
    nids = np.concatenate(batch).reshape(-1).astype(np.int64)
    g1 = g.subgraph(nids)
    g1 = dgl.remove_self_loop(g1)
    g1 = dgl.add_self_loop(g1)
    return g1