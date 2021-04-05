from dglke.dataloader.KGDataset import get_dataset
import torch
from kgeutils.ioutils import ArgParser
import sys
import os
from os.path import join

from time import time
from tqdm import tqdm, trange
from kgeutils.utils import seed_everything, json_to_argv
from dglke.dataloader.KGEDataloader import KGETrainDataset, KGETestDataset
from kgeutils.gpu_utils import device_setting
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train_kge_run():
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
    train_data = KGETrainDataset(dataset=dataset, negative_sample_size=args.neg_sample_size, mode='head-batch')
    test_data = KGETestDataset(dataset=dataset, mode='head-batch')
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    train_kge_run()