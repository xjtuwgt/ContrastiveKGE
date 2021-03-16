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
    seed_everything(seed=args.rand_seed)
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
    loss_in_batchs = []
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        # for key, value in batch.items():
        #     print(key, value)
        anchor = batch['anchor']
        # print(anchor)

        # node_number_in_batchs.append(batch['node_number'])
        # print(batch['batch_graph'].ndata[dgl.NID])

        batch_g = batch['batch_graph']
        loss, cls_embed = model.forward(batch_g)
        loss_in_batchs.append(loss.data.item())
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
    print(max(loss_in_batchs))
    print(min(loss_in_batchs))
    print(sum(loss_in_batchs)/len(loss_in_batchs))
    # print('tid {}'.format(graph.edata['tid'][0]))
    print('Run time {}'.format(time() - start_time))
