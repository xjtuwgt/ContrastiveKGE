# -*- coding: utf-8 -*-
#
# ioutils.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import os
import csv
import argparse
from argparse import ArgumentParser
import json
from os.path import join
import numpy as np
import torch
from envs import KG_DATASET_FOLDER, OUTPUT_FOLDER

def get_compatible_batch_size(batch_size, neg_sample_size):
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print('batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.format(
            old_batch_size, neg_sample_size, batch_size))
    return batch_size

def save_model(args, model, emap_file=None, rmap_file=None):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print('Save model to {}'.format(args.save_path))
    model.save_emb(args.save_path, args.dataset)

    # We need to save the model configurations as well.
    conf_file = os.path.join(args.save_path, 'config.json')
    dict = {}
    config = args
    dict.update(vars(config))
    dict.update({'emp_file': emap_file,
                  'rmap_file': rmap_file})
    with open(conf_file, 'w') as outfile:
        json.dump(dict, outfile, indent=4)

def load_model_config(config_f):
    print(config_f)
    with open(config_f, "r") as f:
        config = json.loads(f.read())
        #config = json.load(f)
    print(config)
    return config

def load_raw_triplet_data(head_f=None, rel_f=None, tail_f=None, emap_f=None, rmap_f=None):
    if emap_f is not None:
        eid_map = {}
        id2e_map = {}
        with open(emap_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                eid_map[row[1]] = int(row[0])
                id2e_map[int(row[0])] = row[1]

    if rmap_f is not None:
        rid_map = {}
        id2r_map = {}
        with open(rmap_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                rid_map[row[1]] = int(row[0])
                id2r_map[int(row[0])] = row[1]

    if head_f is not None:
        head = []
        with open(head_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                head.append(eid_map[id[:-1]])
                id = f.readline()
        head = np.asarray(head)
    else:
        head = None

    if rel_f is not None:
        rel = []
        with open(rel_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                rel.append(rid_map[id[:-1]])
                id = f.readline()
        rel = np.asarray(rel)
    else:
        rel = None

    if tail_f is not None:
        tail = []
        with open(tail_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                tail.append(eid_map[id[:-1]])
                id = f.readline()
        tail = np.asarray(tail)
    else:
        tail = None

    return head, rel, tail, id2e_map, id2r_map

def load_triplet_data(head_f=None, rel_f=None, tail_f=None):
    if head_f is not None:
        head = []
        with open(head_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                head.append(int(id))
                id = f.readline()
        head = np.asarray(head)
    else:
        head = None

    if rel_f is not None:
        rel = []
        with open(rel_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                rel.append(int(id))
                id = f.readline()
        rel = np.asarray(rel)
    else:
        rel = None

    if tail_f is not None:
        tail = []
        with open(tail_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                tail.append(int(id))
                id = f.readline()
        tail = np.asarray(tail)
    else:
        tail = None

    return head, rel, tail

def load_raw_emb_mapping(map_f):
    assert map_f is not None
    id2e_map = {}
    with open(map_f, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            id2e_map[int(row[0])] = row[1]

    return id2e_map


def load_raw_emb_data(file, map_f=None, e2id_map=None):
    if map_f is not None:
        e2id_map = {}
        id2e_map = {}
        with open(map_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                e2id_map[row[1]] = int(row[0])
                id2e_map[int(row[0])] = row[1]
    elif e2id_map is not None:
        id2e_map = [] # dummpy return value
    else:
        assert False, 'There should be an ID mapping file provided'

    ids = []
    with open(file, 'r') as f:
        line = f.readline()
        while len(line) > 0:
            ids.append(e2id_map[line[:-1]])
            line = f.readline()
        ids = np.asarray(ids)

    return ids, id2e_map, e2id_map

def load_entity_data(file=None):
    if file is None:
        return None

    entity = []
    with open(file, 'r') as f:
        id = f.readline()
        while len(id) > 0:
            entity.append(int(id))
            id = f.readline()
    entity = np.asarray(entity)
    return entity


class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE',
                                   'SimplE'],
                          help='The models provided by DGL-KE.')
        self.add_argument('--data_path', type=str, default=KG_DATASET_FOLDER,
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--dataset', type=str, default='wn18rr',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge '\
                                  'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. '\
                                  'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        # # args.dataset = 'FB15k-237'
        # args.dataset = 'wn18rr'
        # # args.dataset = 'FB15k'
        # # args.dataset = 'wn18'
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs,'\
                                  'the foramt should be built_in. For users own knowledge graphs,'\
                                  'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE'\
                                  'on their own datasets. If the format is raw_udd_{htr},'\
                                  'users need to provide train_file [valid_file] [test_file].'\
                                  'If the format is udd_{htr}, users need to provide'\
                                  'entity_file relation_file train_file [valid_file] [test_file].'\
                                  'In both cases, valid_file and test_file are optional.')
        self.add_argument('--delimiter', type=str, default='\t',
                          help='Delimiter used in data files. Note all files should use the same delimiter.')
        self.add_argument('--save_path', type=str, default=OUTPUT_FOLDER,
                          help='the path of the directory where models and logs are saved.')
        self.add_argument('--no_save_emb', action='store_true',
                          help='Disable saving the embeddings under save_path.')
        self.add_argument('--max_step', type=int, default=80000,
                          help='The maximal number of steps to train the model.'\
                                  'A step trains the model with a batch of data.')
        self.add_argument('--batch_size', type=int, default=1024,
                          help='The batch size for training.')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='The batch size used for validation and test.')
        self.add_argument('--neg_sample_size', type=int, default=256,
                          help='The number of negative samples we use for each positive sample in the training.')
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the training.'\
                                  'When this option is turned on, the number of negative samples per positive edge'\
                                  'will be doubled. Half of the negative samples are generated uniformly while'\
                                  'the other half are generated proportional to vertex degree.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the evaluation.')
        self.add_argument('--neg_sample_size_eval', type=int, default=-1,
                          help='The number of negative samples we use to evaluate a positive sample.')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='Randomly sample some percentage of edges for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('-log', '--log_interval', type=int, default=1000,
                          help='Print runtime of different components every x steps.')
        self.add_argument('--eval_interval', type=int, default=10000,
                          help='Print evaluation results on the validation dataset every x steps'\
                                  'if validation is turned on')
        self.add_argument('--test_log_steps', type=int, default=100,  help='print test frequency if validation is turned on')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the test set after the model is trained.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to train the model in parallel.'\
                                  'In multi-GPU training, the number of processes by default is set to match the number of GPUs.'\
                                  'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to train the model in each process.'\
                                  'This argument is used for multiprocessing training.')
        self.add_argument('--force_sync_interval', type=int, default=-1,
                          help='We force a synchronization between processes every x steps for'\
                                  'multiprocessing training. This potentially stablizes the training process'
                                  'to get a better performance. For multiprocessing training, it is set to 1000 by default.')
        # self.add_argument('--hidden_dim', type=int, default=400,
        #                   help='The embedding size of relation and entity')
        # self.add_argument('--lr', type=float, default=1e-3,
        #                   help='The learning rate. DGL-KE uses Adagrad to optimize the model parameters.')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='Double entitiy dim for complex number or canonical polyadic. It is used by RotatE and SimplE.')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='Double relation dim for complex number or canonical polyadic. It is used by RotatE and SimplE')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true',
                          help='Indicate whether to use negative adversarial sampling.'\
                                  'It will weight negative samples with higher scores more.')
        self.add_argument('-a', '--adversarial_temperature', default=1.0, type=float,
                          help='The temperature used for negative adversarial sampling.')
        self.add_argument('-rc', '--regularization_coef', type=float, default=0.000002,
                          help='The coefficient for regularization.')
        self.add_argument('-rn', '--regularization_norm', type=int, default=3,
                          help='norm used in regularization.')
        self.add_argument('-pw', '--pairwise', action='store_true',
                          help='Indicate whether to use pairwise loss function. '
                               'It compares the scores of a positive triple and a negative triple')
        self.add_argument('--loss_genre', default='Logsigmoid',
                          choices=['Hinge', 'Logistic', 'Logsigmoid', 'BCE'],
                          help='The loss function used to train KGEM.')
        self.add_argument('-m', '--margin', type=float, default=1.0,
                          help='hyper-parameter for hinge loss.')


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--do_train', action='store_true', default=True)
        self.add_argument('--do_valid', action='store_true', default=True)
        self.add_argument('--do_test', action='store_true', default=True)
        self.add_argument('--hop_num', type=int, default=2, help='hop_number to generate the sub-graph')
        self.add_argument('--fanouts', default='15,10', type=str,
                          help='fanout, 1-hop number of sample neighbors, 2-hop, 3-hop')
        self.add_argument('--edge_dir', type=str, default='all', help='edge direction to generate the sub-graphs')

        self.add_argument('--graph_batch_size', type=int, default=32, help='batch size for contrastive learning')
        self.add_argument('--dev_graph_batch_size', type=int, default=16, help='batch size for contrastive learning')
        self.add_argument('--cpu_num', type=int, default=16, help='number of cpus for data loader')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'\
                                  'The positive score will be adjusted '\
                                  'as pos_score = pos_score * edge_importance')
        self.add_argument('--add_special', default=True, action='store_true', help='adding special entity/relation')
        self.add_argument('--reverse_r', default=True, action='store_true', help='adding special entity/relation')
        self.add_argument('--ent_dim', type=int, default=512, help='kg embedding dimension')
        self.add_argument('--rel_dim', type=int, default=512, help='kg embedding dimension')
        self.add_argument('--graph_hidden_dim', type=int, default=512, help='graph hidden dimension')
        self.add_argument('--head_num', type=int, default=4, help='head number of GNNs')
        self.add_argument('--attn_drop', type=float, default=0.25, help='attention dropout for GNN')
        self.add_argument('--feat_drop', type=float, default=0.25, help='feature dropout for GNN')
        self.add_argument('--negative_slope', type=float, default=0.4, help='negative slope for elu activation function')
        self.add_argument('--residual', default=True, action='store_true', help='whether adding residual connection')
        self.add_argument('--diff_head_tail', default=True, action='store_true', help='whether distinguish head and tail')
        self.add_argument('--layers', default=2, type=int, help='number of GNN layers')
        self.add_argument('--cuda', default=False, action='store_true', help='use GPU')
        self.add_argument('--gpus', default=-1, type=int, help='use GPU')
        self.add_argument("--gpu", type=int, default=-1,
                            help="which GPU to use. Set -1 to use CPU.")
        self.add_argument("--gpu_id", default=None, type=str, help="GPU id")

        self.add_argument('--rand_seed', default=42, type=int, help='Random seed for initialization')
        self.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

        self.add_argument("--config_file", type=str, default=None, help="configuration file for command parser")
        self.add_argument("--exp_name", type=str, default=None,
                          help="If set, this will be used as directory name in OUTOUT folder")

        self.add_argument("--n_entities", type=int, default=None, help="number of entities")
        self.add_argument("--n_relations", type=int, default=None, help="number of relations")
        self.add_argument("--xent_temperature", type=float, default=0.10, help="temperature of cross entropy")

        self.add_argument('--learning_rate', default=2e-4, type=float, help='Learning rate')
        self.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
        self.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.add_argument('--num_train_epochs', type=int, default=100, help="number of train epochs")
        self.add_argument('--max_steps', type=int, default=80000, help="number of training steps")
        self.add_argument('--warmup_steps', type=int, default=0, help="warm up steps")
        self.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient_accumulation_steps")
        self.add_argument('--eval_interval_ratio', default=0.1, type=float, help='Evaluation interval ratio')
        self.add_argument('--init_checkpoint', type=str, default=None, help="initial check point path and name")
        self.add_argument('--negative_adversarial_sampling', action='store_true', default=True)
        self.add_argument('--uni_weight', action='store_true', default=True, help='Otherwise use subsampling weighting like in word2vec')
        self.add_argument('--save_checkpoint_steps', type=int, default=10000, help='save steps')





def complete_default_train_parser(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # set n_gpu
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.gpus = torch.cuda.device_count()
        else:
            args.gpus = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.gpus = 1
    args.device = device

    # output dir name
    if not args.exp_name:
        args.exp_name = '_'.join(['lr' + str(args.learning_rate),
                          'bs' + str(args.graph_batch_size)])
    args.exp_name = join(args.output_dir, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    torch.save(args, join(args.exp_name, "training_args.bin"))
    return args

def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

def save_args(args, path):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(args_path):
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args