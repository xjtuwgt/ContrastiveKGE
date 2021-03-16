from dglke.dataloader.KGDataset import get_dataset
import torch
from kgeutils.ioutils import ArgParser
from dglke.dataloader.KGDataloader import train_data_loader
from dglke.models.ContrastiveKGEmodels import ContrastiveKEModel
import torch.nn.functional as F

from time import time
from tqdm import tqdm, trange
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
    args.dataset = 'wn18rr'
    # args.dataset = 'FB15k'
    # args.dataset = 'wn18'

    for key, value in vars(args).items():
        logging.info("{}:{}".format(key, value))
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Initial number of entities: {}'.format(dataset.n_entities))
    logging.info('Initial number of relations: {}'.format(dataset.n_relations))
    device = device_setting(args=args)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    tr_data_loader, n_entities, n_relations = train_data_loader(args=args, dataset=dataset)
    logging.info('graph based number of entities: {}'.format(n_entities))
    logging.info('graph based number of relations: {}'.format(n_relations))
    ###++++++++++++++++++++++++++++++++++++++++++
    total_batch_num = len(tr_data_loader)
    logger.info('Total number of batches = {}'.format(total_batch_num))
    eval_batch_interval_num = int(total_batch_num * args.eval_interval_ratio) + 1
    logger.info('Evaluate the model by = {} batches'.format(eval_batch_interval_num))
    ###++++++++++++++++++++++++++++++++++++++++++
    args.n_entities = n_entities
    args.n_relations = n_relations
    model = ContrastiveKEModel(n_relations=args.n_relations, n_entities=args.n_entities, ent_dim=args.ent_dim, rel_dim=args.rel_dim,
                               gamma=args.gamma, activation=F.elu, attn_drop=args.attn_drop, feat_drop=args.feat_drop,
                                          head_num=args.head_num, graph_hidden_dim=args.graph_hid_dim,
                                          n_layers=args.layers)

    model.to(device)
    model.zero_grad()
    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_time = time()
    loss_in_batchs = []
    start_epoch = 0
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        epoch_iterator = tqdm(tr_data_loader, desc="Iteration", miniters=200, disable=args.local_rank not in [-1, 0])
        for batch_idx, batch in enumerate(epoch_iterator):
            for key, value in batch.items():
                batch[key] = value.to(device)
            model.train()
            batch_g = batch['batch_graph']
            loss, cls_embed = model.forward(batch_g)
            loss_in_batchs.append(loss.data.item())
            del batch
            # break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            model.zero_grad()
            if (batch_idx + 1) % eval_batch_interval_num == 0:
                logging.info("Epoch {:05d} | Step {:05d} | Time(s) {:.4f} | Loss {:.4f}"
                             .format(epoch + 1, batch_idx +1, time() - start_time, loss.item()))

    print(max(loss_in_batchs))
    print(min(loss_in_batchs))
    print(sum(loss_in_batchs)/len(loss_in_batchs))
    # print('tid {}'.format(graph.edata['tid'][0]))
    print('Run time {}'.format(time() - start_time))
