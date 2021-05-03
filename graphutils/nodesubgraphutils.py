from dgl.sampling import sample_neighbors, randomwalks
from dgl import DGLHeteroGraph
from collections import OrderedDict
import itertools
import torch
import dgl
from torch.utils.data import Dataset
from copy import deepcopy

def direct_sub_graph(anchor_node_ids, cls_node_ids, fanouts, g, edge_dir):
    """
    :param anchor_node_ids: a single node idxes
    :param cls_node_ids:
    :param fanouts: size = hop_number
    :param g:
    :param edge_dir: edge direction: in or out
    :return:
    """
    neighbors_dict = {'anchor': anchor_node_ids}
    neighbors_dict['cls'] = cls_node_ids
    edge_dict = {}
    hop = 1
    while hop < len(fanouts) + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['hop_{}'.format(hop - 1)]
        sg = sample_neighbors(g=g, nodes=node_ids, edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata['_ID'], sg.edata['tid'] # tid: type ids
        sg_src_list, sg_dst_list = sg_src.tolist(), sg_dst.tolist()
        sg_eid_list, sg_tid_list = sg_eids.tolist(), sg_tids.tolist()
        for eid, src_id, tid, dst_id in zip(sg_eid_list, sg_src_list, sg_tid_list, sg_dst_list):
            edge_dict[eid] = (src_id, tid, dst_id)
        if edge_dir == 'in':
            hop_neighbor = sg_src
        elif edge_dir == 'out':
            hop_neighbor = sg_dst
        else:
            raise 'Edge direction {} is not supported'.format(edge_dir)
        neighbors_dict['hop_{}'.format(hop)] = hop_neighbor
        hop = hop + 1
    return neighbors_dict, edge_dict

def sub_graph_extractor(g: DGLHeteroGraph, neighbor_dict_pair: tuple, edge_dict_pair: tuple,
                        edge_dir: str, n_relations: int, cls_id: int,
                        special_relation2id: dict, reverse=False):
    """
    :param g: graph
    :param neighbor_dict_pair: (in neighbors, out neighbors)
    :param edge_dict_pair: (in edge set, out edge set)
    :param edge_dir: in or out
    :param n_relations: number of relations
    :param cls_id: (cls id)
    :param special_relation2id: relation to relation ids
    :param reverse: whether adding reverse edges
    :return:
    """
    in_neighbor_dict, out_neighbor_dict = neighbor_dict_pair
    in_edge_dict, out_edge_dict = edge_dict_pair
    if edge_dir == 'in':
        sub_graph_node_tensor_list = [in_neighbor_dict['cls'], in_neighbor_dict['anchor']]
        sub_graph_node_tensor_list += [value for key, value in in_neighbor_dict.items() if key not in ['cls', 'anchor']]
        sub_graph_nodes = torch.cat(sub_graph_node_tensor_list).tolist()
        sub_graph_edge_dict = in_edge_dict
    elif edge_dir == 'out':
        sub_graph_node_tensor_list = [out_neighbor_dict['cls'], out_neighbor_dict['anchor']]
        sub_graph_node_tensor_list += [value for key, value in out_neighbor_dict.items() if key not in ['cls', 'anchor']]
        sub_graph_nodes = torch.cat(sub_graph_node_tensor_list).tolist()
        sub_graph_edge_dict = out_edge_dict
    elif edge_dir == 'all':
        sub_graph_node_tensor_list = [in_neighbor_dict['cls'], in_neighbor_dict['anchor']]
        sub_graph_node_tensor_list += [value for key, value in in_neighbor_dict.items() if key not in ['cls', 'anchor']]
        sub_graph_node_tensor_list += [value for key, value in out_neighbor_dict.items() if key not in ['cls', 'anchor']]
        sub_graph_nodes = torch.cat(sub_graph_node_tensor_list).tolist()
        sub_graph_edge_dict = {**in_edge_dict, **out_edge_dict}
    else:
        raise 'Edge direction {} is not supported'.format(edge_dir)
    sub_graph_nodes = list(OrderedDict.fromkeys(sub_graph_nodes))
    # print(len(sub_graph_edge_dict), len(sub_graph_nodes))
    # print(len(sub_graph_nodes), len(set(sub_graph_nodes)))
    sub_graph = dgl.node_subgraph(graph=g, nodes=sub_graph_nodes)
    sub_graph.ndata['nid'] = sub_graph.ndata[dgl.NID]
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # print('subgraph', sub_graph.number_of_edges(), sub_graph.number_of_nodes())
    # print('*' * 75)
    if reverse:
        sg_src, sg_dst = sub_graph.edges()
        sub_graph.add_edges(u=sg_dst, v=sg_src, data={'tid': sub_graph.edata['tid'] + n_relations})
    ## adding cls relation
    cls_id_idx = (sub_graph.ndata['nid'] == cls_id).nonzero(as_tuple=True)[0]
    assert cls_id_idx == 0 ## the first id in
    node_ids = torch.arange(1, sub_graph.number_of_nodes())
    cls_dst = torch.empty(sub_graph.number_of_nodes() - 1, dtype=torch.long).fill_(cls_id_idx[0])
    cls_rel = torch.empty(sub_graph.number_of_nodes() - 1, dtype=torch.long).fill_(special_relation2id['cls_r'])
    sub_graph.add_edges(u=node_ids, v=cls_dst, data={'tid': cls_rel})
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    node_ids = torch.arange(0, sub_graph.number_of_nodes())
    loop_rel = torch.empty(sub_graph.number_of_nodes(), dtype=torch.long).fill_(special_relation2id['loop_r'])
    sub_graph.add_edges(u=node_ids, v=node_ids, data={'tid': loop_rel})
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return sub_graph, cls_id_idx

def dense_graph_constructor(neighbor_dict_pair: tuple, sub_graph, hop_num, special_relation2id: dict, edge_dir: str, reverse=False):
    dense_sub_graph = deepcopy(sub_graph)
    in_neighbor_dict, out_neighbor_dict = neighbor_dict_pair
    sub_graph_par_ids = sub_graph.ndata['nid'].tolist()
    par_to_son_map = dict(zip(sub_graph_par_ids, list(range(len(sub_graph_par_ids)))))

    if edge_dir == 'in':
        anchor_id = in_neighbor_dict['anchor'][0]
        anchor_sub_id = par_to_son_map[anchor_id.data.item()]
        for hop in range(1, hop_num + 1):
            hop_key = 'hop_{}'.format(hop)
            hop_i_nodes = in_neighbor_dict[hop_key].tolist()
            hop_i_sub_ids = [par_to_son_map[_] for _ in hop_i_nodes]
            hop_rel = torch.empty(len(hop_i_nodes), dtype=torch.long).fill_(special_relation2id[hop_key])
            dense_sub_graph.add_edges(u=hop_i_sub_ids, v=[anchor_sub_id] * len(hop_i_nodes), data={'tid': hop_rel})
            if reverse:
                rev_hop_key = 'rev_hop_{}'.format(hop)
                rev_hop_rel = torch.empty(len(hop_i_nodes), dtype=torch.long).fill_(special_relation2id[rev_hop_key])
                dense_sub_graph.add_edges(u=[anchor_sub_id] * len(hop_i_nodes), v=hop_i_sub_ids, data={'tid': rev_hop_rel})
    elif edge_dir == 'out':
        anchor_id = out_neighbor_dict['anchor'][0]
        anchor_sub_id = par_to_son_map[anchor_id.data.item()]
        for hop in range(1, hop_num + 1):
            hop_key = 'hop_{}'.format(hop)
            hop_i_nodes = out_neighbor_dict[hop_key].tolist()
            hop_i_sub_ids = [par_to_son_map[_] for _ in hop_i_nodes]
            hop_rel = torch.empty(len(hop_i_nodes), dtype=torch.long).fill_(special_relation2id[hop_key])
            dense_sub_graph.add_edges(u=[anchor_sub_id] * len(hop_i_nodes), v=hop_i_sub_ids, data={'tid': hop_rel})
            if reverse:
                rev_hop_key = 'rev_hop_{}'.format(hop)
                rev_hop_rel = torch.empty(len(hop_i_nodes), dtype=torch.long).fill_(special_relation2id[rev_hop_key])
                dense_sub_graph.add_edges(u=hop_i_sub_ids, v=[anchor_sub_id] * len(hop_i_nodes), data={'tid': rev_hop_rel})
    elif edge_dir == 'all':
        anchor_id = out_neighbor_dict['anchor'][0]
        anchor_sub_id = par_to_son_map[anchor_id.data.item()]
        for hop in range(1, hop_num + 1):
            hop_key = 'hop_{}'.format(hop)
            in_hop_i_nodes = in_neighbor_dict[hop_key].tolist()
            in_hop_i_sub_ids = [par_to_son_map[_] for _ in in_hop_i_nodes]
            in_hop_rel = torch.empty(len(in_hop_i_nodes), dtype=torch.long).fill_(special_relation2id[hop_key])
            dense_sub_graph.add_edges(u=in_hop_i_sub_ids, v=[anchor_sub_id] * len(in_hop_i_nodes), data={'tid': in_hop_rel})
            if reverse:
                rev_hop_key = 'rev_hop_{}'.format(hop)
                in_rev_hop_rel = torch.empty(len(in_hop_i_nodes), dtype=torch.long).fill_(special_relation2id[rev_hop_key])
                dense_sub_graph.add_edges(u=in_hop_i_sub_ids, v=[anchor_sub_id] * len(in_hop_i_nodes),
                                          data={'tid': in_rev_hop_rel})
            ####################################################################################
            out_hop_i_nodes = out_neighbor_dict[hop_key].tolist()
            out_hop_i_sub_ids = [par_to_son_map[_] for _ in out_hop_i_nodes]
            out_hop_rel = torch.empty(len(out_hop_i_nodes), dtype=torch.long).fill_(special_relation2id[hop_key])
            dense_sub_graph.add_edges(u=[anchor_sub_id] * len(out_hop_i_nodes), v=out_hop_i_sub_ids, data={'tid': out_hop_rel})
            if reverse:
                rev_hop_key = 'rev_hop_{}'.format(hop)
                out_rev_hop_rel = torch.empty(len(out_hop_i_nodes), dtype=torch.long).fill_(special_relation2id[rev_hop_key])
                dense_sub_graph.add_edges(u=out_hop_i_sub_ids, v=[anchor_sub_id] * len(out_hop_i_nodes),
                                          data={'tid': out_rev_hop_rel})
    else:
        raise 'Edge direction {} is not supported'.format(edge_dir)
    anchor_sub_id = torch.LongTensor([anchor_sub_id])
    return dense_sub_graph, anchor_sub_id


class SubGraphPairDataset(Dataset):
    def __init__(self, g: DGLHeteroGraph, nentity: int, nrelation: int,
                 fanouts: list, special_entity2id: dict, special_relation2id: dict,
                 replace=False, reverse=False, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts
        self.hop_num = len(fanouts)
        self.g = g
        #####################
        if len(special_entity2id) > 0:
            self.len = g.number_of_nodes() - len(special_entity2id)  ## no need to extract sub-graph of special entities
        else:
            self.len = g.number_of_nodes()
        #####################
        self.nentity = nentity
        self.nrelation = nrelation
        self.reverse = reverse
        self.fanouts = fanouts ## list of int == number of hops for sampling
        self.edge_dir = edge_dir ## "in", "out", "all"
        self.replace = replace
        self.special_entity2id = special_entity2id
        self.special_relation2id = special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        if self.edge_dir == 'in':
            in_neighbors_dict, in_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
                                             g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
            out_neighbors_dict, out_edge_dict = None, None
        elif self.edge_dir == 'out':
            in_neighbors_dict, in_edge_dict = None, None
            out_neighbors_dict, out_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
                                               g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
        elif self.edge_dir == 'all':
            in_neighbors_dict, in_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
                                             g=self.g, fanouts=self.fanouts, edge_dir='in')
            out_neighbors_dict, out_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids, cls_node_ids=cls_node_ids,
                                               g=self.g, fanouts=self.fanouts, edge_dir='out')
        else:
            raise 'Edge direction {} is not supported'.format(self.edge_dir)

        sub_graph, cls_sub_id = sub_graph_extractor(g=self.g, neighbor_dict_pair=(in_neighbors_dict, out_neighbors_dict),
                                                    edge_dict_pair=(in_edge_dict, out_edge_dict),
                                               cls_id=self.special_entity2id['cls'], special_relation2id=self.special_relation2id,
                                               edge_dir=self.edge_dir, reverse=self.reverse, n_relations=self.nrelation)
        dense_sub_graph, anchor_sub_id = dense_graph_constructor(neighbor_dict_pair=(in_neighbors_dict, out_neighbors_dict), sub_graph=sub_graph,
                                                  hop_num=self.hop_num,
                                                  special_relation2id=self.special_relation2id, reverse=self.reverse, edge_dir=self.edge_dir)
        return anchor_node_ids, cls_sub_id, anchor_sub_id, sub_graph, dense_sub_graph

    @staticmethod
    def collate_fn(data):
        anchor_nodes = torch.cat([_[0] for _ in data], dim=0)
        cls_sub_ids = torch.cat([_[1] for _ in data], dim=0)
        anchor_sub_ids = torch.cat([_[2] for _ in data], dim=0)
        batch_graphs = dgl.batch(list(itertools.chain.from_iterable([(_[3], _[4]) for _ in data])))

        number_of_nodes = torch.LongTensor([sum([_[3].number_of_nodes() for _ in data])])[0]
        sparse_number_of_edges = torch.LongTensor([sum([_[3].number_of_edges() for _ in data])])[0]
        dense_number_of_edges = sum([_[4].number_of_edges() for _ in data])
        edge_number = torch.LongTensor([sparse_number_of_edges + dense_number_of_edges])[0]
        batch = {'anchor': anchor_nodes, 'cls': cls_sub_ids, 'anchor_sub': anchor_sub_ids,
                 'node_number': number_of_nodes, 'edge_number': edge_number, 'batch_graph': batch_graphs}
        return batch

class SubGraphDataset(Dataset):
    def __init__(self, g: DGLHeteroGraph, nentity: int, nrelation: int,
                 fanouts: list, special_entity2id: dict, special_relation2id: dict,
                 replace=False, reverse=False, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts
        self.hop_num = len(fanouts)
        self.g = g
        #####################
        if len(special_entity2id) > 0:
            self.len = g.number_of_nodes() - len(special_entity2id) ## no need to extract sub-graph of special entities
        else:
            self.len = g.number_of_nodes()
        #####################
        self.nentity = nentity
        self.nrelation = nrelation
        self.reverse = reverse
        self.fanouts = fanouts ## list of int == number of hops for sampling
        self.edge_dir = edge_dir ## "in", "out", "all"
        self.replace = replace
        self.special_entity2id = special_entity2id
        self.special_relation2id = special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        if self.edge_dir == 'in':
            in_neighbors_dict, in_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids,
                                                               cls_node_ids=cls_node_ids,
                                                               g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
            out_neighbors_dict, out_edge_dict = None, None
        elif self.edge_dir == 'out':
            in_neighbors_dict, in_edge_dict = None, None
            out_neighbors_dict, out_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids,
                                                                 cls_node_ids=cls_node_ids,
                                                                 g=self.g, fanouts=self.fanouts, edge_dir=self.edge_dir)
        elif self.edge_dir == 'all':
            in_neighbors_dict, in_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids,
                                                               cls_node_ids=cls_node_ids,
                                                               g=self.g, fanouts=self.fanouts, edge_dir='in')
            out_neighbors_dict, out_edge_dict = direct_sub_graph(anchor_node_ids=anchor_node_ids,
                                                                 cls_node_ids=cls_node_ids,
                                                                 g=self.g, fanouts=self.fanouts, edge_dir='out')
        else:
            raise 'Edge direction {} is not supported'.format(self.edge_dir)

        sub_graph, cls_sub_id = sub_graph_extractor(g=self.g, neighbor_dict_pair=(in_neighbors_dict, out_neighbors_dict),
                                                    edge_dict_pair=(in_edge_dict, out_edge_dict),
                                                cls_id=self.special_entity2id['cls'], special_relation2id=self.special_relation2id,
                                               edge_dir=self.edge_dir, reverse=self.reverse, n_relations=self.nrelation)
        sub_graph_par_ids = sub_graph.ndata['nid'].tolist()
        par_to_son_map = dict(zip(sub_graph_par_ids, list(range(len(sub_graph_par_ids)))))
        anchor_sub_id = par_to_son_map[idx]
        anchor_sub_id = torch.LongTensor([anchor_sub_id])
        return anchor_node_ids, cls_sub_id, anchor_sub_id, sub_graph

    @staticmethod
    def collate_fn(data):
        anchor_nodes = torch.cat([_[0] for _ in data], dim=0)
        cls_sub_ids = torch.cat([_[1] for _ in data], dim=0)
        anchor_sub_ids = torch.cat([_[2] for _ in data], dim=0)
        batch_graphs = dgl.batch([_[3] for _ in data])

        number_of_nodes = torch.LongTensor([sum([_[3].number_of_nodes() for _ in data])])[0]
        number_of_edges = torch.LongTensor([sum([_[3].number_of_edges() for _ in data])])[0]
        edge_number = torch.LongTensor([number_of_edges])[0]
        batch = {'anchor': anchor_nodes, 'cls': cls_sub_ids, 'anchor_sub': anchor_sub_ids,
                 'node_number': number_of_nodes, 'edge_number': edge_number, 'batch_graph': batch_graphs}
        return batch