from dglke.dataloader.KGDataset import get_dataset
from dglke.dataloader.KGutils import ConstructGraph
from dglke.dataloader.KGDataloader import TrainDataset, EvalDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import dgl
from dgl.contrib.sampling import EdgeSampler
from dglke.dataloader.KGutils import SoftRelationPartition, RandomPartition
from dglke.ioutils import CommonArgParser
from dglke.dataloader.KGDataloader import train_data_loader
from dglke.models.ContrastiveKGEmodels import ContrastiveKEModel
from dglke.models.ContrastiveKGEmodels import GraphContrastiveLoss
from dgl.dataloading.pytorch import EdgeCollator
from dglke.models.kgemodels import KEModel
from copy import deepcopy
from graphutils.gsampleutils import SubGraphPairDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from figureUtils.plotUtils import distribution_plot
from time import time
from tqdm import tqdm
from pytorch_metric_learning.losses import NTXentLoss


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--hop_num', type=int, default=2, help='hop_number to generate the sub-graph')
        self.add_argument('--edge_dir', type=str, default='all', help='edge direction to generate the sub-graphs')
        self.add_argument('--graph_batch_size', type=int, default=8, help='batch size for contrastive learning')
        self.add_argument('--cpu_num', type=int, default=8, help='number of cpus for data loader')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'\
                                  'The positive score will be adjusted '\
                                  'as pos_score = pos_score * edge_importance')
        self.add_argument('--add_special', default=True, action='store_true', help='adding special entity/relation')
        self.add_argument('--fanouts', default='15,10', type=str, help='fanout, 1-hop number of sample neighbors, 2-hop, 3-hop')
        self.add_argument('--reverse_r', default=True, action='store_true', help='adding special entity/relation')
        self.add_argument('--ent_dim', type=int, default=256, help='kg embedding dimension')
        self.add_argument('--rel_dim', type=int, default=256, help='kg embedding dimension')
        self.add_argument('--graph_hid_dim', type=int, default=256, help='graph hidden dimension')
        self.add_argument('--head_num', type=int, default=4, help='head number of GNNs')
        self.add_argument('--attn_drop', type=float, default=0.25, help='attention dropout for GNN')
        self.add_argument('--feat_drop', type=float, default=0.25, help='feature dropout for GNN')
        self.add_argument('--negative_slope', type=float, default=0.4, help='negative slope for elu activation function')
        self.add_argument('--residual', default=True, action='store_true', help='whether adding residual connection')
        self.add_argument('--diff_head_tail', default=True, action='store_true', help='whether distinguish head and tail')
        self.add_argument('--layers', default=2, type=int, help='number of GNN layers')
        self.add_argument('--random_seed', default=42, type=int, help='Random seed for initialization')



def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

if __name__ == '__main__':
    args = ArgParser().parse_args()
    for key, value in vars(args).items():
        print("{}:{}".format(key, value))
    # args.dataset = 'FB15k-237'
    args.dataset = 'wn18rr'
    # args.dataset = 'FB15k'
    # args.dataset = 'wn18'

    # hop_num = 4
    # hop_num = 2
    hop_num = args.hop_num
    edge_dir = args.edge_dir
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)

    train_data = TrainDataset(dataset=dataset, hop_num=args.hop_num, add_special=args.add_special,
                              reverse=args.reverse_r,
                              has_importance=args.has_edge_importance)
    graph = train_data.g

    print(dataset.n_entities)
    data_loader, n_entities, n_relations = train_data_loader(args=args, dataset=dataset)
    model = ContrastiveKEModel(n_relations=n_relations, n_entities=n_entities, ent_dim=args.ent_dim, rel_dim=args.rel_dim,
                               gamma=args.gamma, activation=F.elu, attn_drop=args.attn_drop, feat_drop=args.feat_drop,
                                          head_num=args.head_num, graph_hidden_dim=args.graph_hid_dim,
                                          n_layers=args.layers)
    start_time = time()
    node_number_in_batchs = []
    sparse_edge_number_in_batchs = []
    dense_edge_number_in_batchs = []
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        # for key, value in batch.items():
        #     print(key, value)
        anchor = batch['anchor']
        # print(anchor)

        # node_number_in_batchs.append(batch['node_number'])
        # print(batch['batch_graph'].ndata[dgl.NID])

        batch_g = batch['batch_graph']
        loss, cls_embed = model.forward(batch_g)
        # print(loss)
        # print(cls_embed.shape)
        # x = dgl.unbatch(batch_g)
        # print(batch_g.number_of_edges())
        # print(batch['edge_number'])
        # for idx, x_ in enumerate(x):
        #     print(x_.in_degrees(0), x_.number_of_nodes(), x_.out_degrees(0))
        #     print(x_.edges())
        #     # print(idx, x_.ndata[dgl.NID][0])
        #     # print(idx, x_.ndata[dgl.NID][1])
        #
        #     print(x_.edata['tid'])
        #
        #
        #
        #     y = x_.edata[dgl.EID].tolist()
        #     # print(x_.edata)
        #     z = [graph.edata['tid'][_].data.item() for _ in y]
        #     print(torch.LongTensor(z))

        # print(len(x))
        # break

    print('tid {}'.format(graph.edata['tid'][0]))
    print('Run time {}'.format(time() - start_time))
