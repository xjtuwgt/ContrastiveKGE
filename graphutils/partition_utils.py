from time import time
import dgl
from dgl.transform import metis_partition
from dgl import backend as F

def get_partition_list(g, psize: int):
    start_time = time()
    p_gs = metis_partition(g, psize)
    graphs = []
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        nids = F.asnumpy(nids)
        graphs.append(nids)
    print('Graph partition takes {}'.format(time() - start_time))
    return graphs