import dgl
import numpy as np
import scipy as sp
import dgl.backend as F
from dglke.dataloader.KGutils import SoftRelationPartition, RandomPartition, ConstructGraph
from graphutils.edgesubgraphutils import SubGraphPairDataset, SubGraphDataset
from torch.utils.data import DataLoader

class UniformNegativeSampler(object):
    def __init__(self, g, k):
        self.g = g
        self.neg_sampler = dgl.dataloading.negative_sampler.Uniform(k)

    def __call__(self, eids):
        return self.neg_sampler(self.g, eids=eids)

class PowerLawInDegNegativeSampler(object):
    def __init__(self, g, k):
        """
        :param g: original graph
        :param k:
        """
        self.weights = g.in_degrees().float() ** 0.75
        self.g = g
        self.k = k

    def __call__(self, eids):
        src, _ = self.g.find_edges(eids)
        src = src.repeat_interleave(self.k)
        dst = self.weights.multinomial(len(src), replacement=True)
        return src, dst

class KGraphDataset(object):
    def __init__(self, dataset, hop_num=0, add_special=False, reverse=False, has_importance=False):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)
        self.n_entities = dataset.n_entities
        self.n_relations = dataset.n_relations
        self.reverse = reverse
        self.hop_num = hop_num
        self.add_special = add_special
        self.has_importance = has_importance
        self.g, self.special_entity_dict, self.special_relation_dict = \
            ConstructGraph(edges=triples, n_entities=dataset.n_entities, n_relations=dataset.n_relations, reverse=reverse,
                                has_edge_importance=has_importance, add_special=add_special, hop_num=hop_num)
        # print(dataset.n_entities)
        # print(self.g.number_of_nodes())
        self.n_relations = self.n_relations + len(self.special_relation_dict)
        self.n_entities = self.n_entities + len(self.special_entity_dict)

class EvalDataset(object):
    def __init__(self, dataset, eval_percent):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        if dataset.valid is not None:
            src.append(dataset.valid[0])
            etype_id.append(dataset.valid[1])
            dst.append(dataset.valid[2])
            self.num_valid = len(dataset.valid[0])
        else:
            self.num_valid = 0
        if dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0
        assert len(src) > 1, "we need to have at least validation set or test set."
        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                   shape=[dataset.n_entities, dataset.n_entities])
        g = dgl.from_scipy(coo)  ## 0.6.2 New graph construction
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        self.g = g

        if eval_percent < 1:
            self.valid = np.random.randint(0, self.num_valid,
                                           size=(int(self.num_valid * eval_percent),)) + self.num_train
        else:
            self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

        if eval_percent < 1:
            self.test = np.random.randint(0, self.num_test,
                                          size=(int(self.num_test * eval_percent, )))
            self.test += self.num_train + self.num_valid
        else:
            self.test = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        print('|test|:', len(self.test))

    def get_edges(self, eval_type):
        """ Get all edges in this dataset
        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

def train_data_loader(args, dataset):
    fanouts = [int(_.strip()) for _ in args.fanouts.split(',')]
    assert len(fanouts) == args.hop_num
    train_graph_data = KGraphDataset(dataset=dataset, hop_num=args.hop_num, add_special=args.add_special, reverse=args.reverse_r,
                              has_importance=args.has_edge_importance)
    sub_graph_pair_data = SubGraphPairDataset(g=train_graph_data.g, nentity=train_graph_data.n_entities,
                                              nrelation=train_graph_data.n_relations,
                                              fanouts=fanouts, special_entity2id=train_graph_data.special_entity_dict,
                                              special_relation2id=train_graph_data.special_relation_dict, edge_dir=args.edge_dir)
    data_loader = DataLoader(dataset=sub_graph_pair_data, batch_size=args.graph_batch_size,
                             shuffle=True,
                             drop_last=True,
                             collate_fn=SubGraphPairDataset.collate_fn, num_workers=args.cpu_num)
    n_entities, n_relations = train_graph_data.n_entities, train_graph_data.n_relations
    return data_loader, n_entities, n_relations

def inference_data_loader(args, dataset):
    assert args.hop_num > 0
    fanouts = [-1] * args.hop_num
    train_graph_data = KGraphDataset(dataset=dataset, hop_num=args.hop_num, add_special=args.add_special,
                              reverse=args.reverse_r,
                              has_importance=args.has_edge_importance)
    sub_graph_data = SubGraphDataset(g=train_graph_data.g, nentity=train_graph_data.n_entities,
                                              nrelation=train_graph_data.n_relations,
                                              fanouts=fanouts, special_entity2id=train_graph_data.special_entity_dict,
                                              special_relation2id=train_graph_data.special_relation_dict,
                                              edge_dir=args.edge_dir)
    data_loader = DataLoader(dataset=sub_graph_data, batch_size=args.dev_graph_batch_size,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=SubGraphDataset.collate_fn, num_workers=args.cpu_num)
    n_entities, n_relations = train_graph_data.n_entities, train_graph_data.n_relations
    return data_loader, n_entities, n_relations
