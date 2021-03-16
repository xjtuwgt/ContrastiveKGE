from dglke.dataloader.KGDataset import get_dataset
from dglke.dataloader.KGDataloader import TrainDataset
import os
from kgeutils.ioutils import ArgParser
from dglke.dataloader.KGDataloader import train_data_loader
from dglke.models.ContrastiveKGEmodels import ContrastiveKEModel
import torch.nn.functional as F

from time import time
from tqdm import tqdm
from kgeutils.utils import seed_everything
from kgeutils.gpu_utils import device_setting
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def run():
    args = ArgParser().parse_args()
    seed_everything(seed=args.rand_seed + args.local_rank)
    # args.dataset = 'FB15k-237'
    # args.dataset = 'wn18rr'
    # args.dataset = 'FB15k'
    args.dataset = 'wn18'

    for key, value in vars(args).items():
        logging.info("{}:{}".format(key, value))
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    hop_num = args.hop_num
    edge_dir = args.edge_dir
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_data = TrainDataset(dataset=dataset, hop_num=args.hop_num, add_special=args.add_special,
                              reverse=args.reverse_r,
                              has_importance=args.has_edge_importance)
    logging.info('Initial number of entities: {}'.format(dataset.n_entities))
    logging.info('Initial number of relations: {}'.format(dataset.n_relations))
    device = device_setting(args=args)
    print(device)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    data_loader, n_entities, n_relations = train_data_loader(args=args, dataset=dataset)
    logging.info('graph based number of entities: {}'.format(n_entities))
    logging.info('graph based number of relations: {}'.format(n_relations))
    args.n_entities = n_entities
    args.n_relations = n_relations
    model = ContrastiveKEModel(n_relations=args.n_relations, n_entities=args.n_entities, ent_dim=args.ent_dim, rel_dim=args.rel_dim,
                               gamma=args.gamma, activation=F.elu, attn_drop=args.attn_drop, feat_drop=args.feat_drop,
                                          head_num=args.head_num, graph_hidden_dim=args.graph_hid_dim,
                                          n_layers=args.layers)
    model.to(device)
    start_time = time()
    loss_in_batchs = []

    for batch_idx, batch in tqdm(enumerate(data_loader)):
        for key, value in batch.items():
            batch[key] = value.to(device)
        batch_g = batch['batch_graph']
        loss, cls_embed = model.forward(batch_g)
        loss_in_batchs.append(loss.data.item())
        # break
    print(max(loss_in_batchs))
    print(min(loss_in_batchs))
    print(sum(loss_in_batchs)/len(loss_in_batchs))
    # print('tid {}'.format(graph.edata['tid'][0]))
    print('Run time {}'.format(time() - start_time))
