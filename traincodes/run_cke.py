from dglke.dataloader.KGDataset import get_dataset
import torch
from kgeutils.ioutils import ArgParser
from dglke.dataloader.KGDataloader import train_data_loader, develop_data_loader
from dglke.models.ContrastiveKGEmodels import ContrastiveKEModel
import sys
import os
from os.path import join

from time import time
from tqdm import tqdm, trange
from kgeutils.utils import seed_everything, get_linear_schedule_with_warmup, json_to_argv
from kgeutils.gpu_utils import device_setting
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train_run():
    parser = ArgParser()
    logger.info("IN CMD MODE")
    args_config_provided = parser.parse_args(sys.argv[1:])
    if args_config_provided.config_file is not None:
        argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
    else:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    seed_everything(seed=args.rand_seed + args.local_rank)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.data_path and not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.exp_name is not None:
        args.exp_name = os.path.join(args.save_path, args.exp_name)
        os.makedirs(args.exp_name, exist_ok=True)
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
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(tr_data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(tr_data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    logger.info('Total training steps = {}'.format(t_total))
    ###++++++++++++++++++++++++++++++++++++++++++
    args.n_entities = n_entities
    args.n_relations = n_relations
    model = ContrastiveKEModel(n_relations=args.n_relations, n_entities=args.n_entities, ent_dim=args.ent_dim, rel_dim=args.rel_dim,
                               gamma=args.gamma, activation=None, attn_drop=args.attn_drop, feat_drop=args.feat_drop,
                                          head_num=args.head_num, graph_hidden_dim=args.graph_hid_dim,
                                          n_layers=args.layers)

    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    logging.info('*' * 75)

    model.to(device)
    model.zero_grad()
    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # use scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    start_time = time()
    loss_in_batchs = []
    start_epoch = 0
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    for epoch in train_iterator:
        epoch_iterator = tqdm(tr_data_loader, desc="Iteration", miniters=100, disable=args.local_rank not in [-1, 0])
        for batch_idx, batch in enumerate(epoch_iterator):
            for key, value in batch.items():
                batch[key] = value.to(device)
            # model.train()

            batch_graph = batch['batch_graph']
            print('anchor', batch['anchor'])
            print('edge num', batch_graph.number_of_edges())
            if batch_idx > 10:
                return
    #         cls_embed = model.forward(batch_graph)
    #         loss = model.loss_computation(cls_embed=cls_embed)
    #         loss_in_batchs.append(loss.data.item())
    #         del batch
    #         # break
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #
    #         optimizer.step()
    #         scheduler.step()
    #         model.zero_grad()
    #         if (batch_idx + 1) % eval_batch_interval_num == 0:
    #             logging.info("Epoch {:05d} | Step {:05d} | Time(s) {:.4f} | Loss {:.4f}"
    #                          .format(epoch + 1, batch_idx +1, time() - start_time, loss.item()))
    # # print('tid {}'.format(graph.edata['tid'][0]))
    # torch.save({k: v.cpu() for k, v in model.state_dict().items()},
    #            join(args.exp_name, f'model.pkl'))
    #
    #
    #
    # print('Run time {}'.format(time() - start_time))


def infer_run():
    parser = ArgParser()
    logger.info("IN CMD MODE")
    args_config_provided = parser.parse_args(sys.argv[1:])
    if args_config_provided.config_file is not None:
        argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
    else:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    seed_everything(seed=args.rand_seed + args.local_rank)

    for key, value in vars(args).items():
        logging.info("{}:{}".format(key, value))
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.data_path and not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
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
    dev_data_loader, n_entities, n_relations = develop_data_loader(args=args, dataset=dataset)
    logging.info('graph based number of entities: {}'.format(n_entities))
    logging.info('graph based number of relations: {}'.format(n_relations))
    ###++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    epoch_iterator = tqdm(dev_data_loader, desc="Iteration", miniters=100, disable=args.local_rank not in [-1, 0])

    for batch_idx, batch in enumerate(epoch_iterator):
        x = batch['anchor']
        # print(batch['node_number'])
        if batch['node_number'] == 1:
            print(batch['batch_graph'])
            print(batch['anchor'])
            print(batch['cls'])
            print(batch['batch_graph'].number_of_nodes())
            print(batch['node_number'])
    print('Run time {}'.format(time() - start_time))