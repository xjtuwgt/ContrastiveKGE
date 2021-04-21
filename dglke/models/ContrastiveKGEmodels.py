from torch import nn
import dgl
import torch
from torch import Tensor as T
from dglke.models.kgembedder import ExternalEmbedding
from dglke.models.GNNmodels import KGELayer
from dgl.nn.pytorch.utils import Identity
from pytorch_metric_learning.losses import NTXentLoss
EMB_INIT_EPS = 2.0

class KGEGraphEncoder(nn.Module):
    def __init__(self, num_layers: int, in_ent_dim: int, in_rel_dim: int, hidden_dim: int, head_num: int,
                feat_drop: float, attn_drop: float, negative_slope=0.2,
                 residual=False, diff_head_tail=False, activation=None):
        super(KGEGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(KGELayer(in_ent_feats=in_ent_dim,
                                        in_rel_feats=in_rel_dim,
                                        out_ent_feats=hidden_dim,
                                        num_heads=head_num,
                                        feat_drop=feat_drop,
                                        attn_drop=attn_drop,
                                        negative_slope=negative_slope,
                                        residual=residual,
                                        activation=activation,
                                        diff_head_tail=diff_head_tail))
        # hidden layers
        for l in range(1, num_layers):
            self.gat_layers.append(KGELayer(in_ent_feats=hidden_dim,
                                            in_rel_feats=in_rel_dim,
                                            out_ent_feats=hidden_dim,
                                            num_heads=head_num,
                                            feat_drop=feat_drop,
                                            attn_drop=attn_drop,
                                            activation=activation,
                                            residual=residual,
                                            negative_slope=negative_slope,
                                            diff_head_tail=diff_head_tail))

    def forward(self, batch_g, ent_embed: ExternalEmbedding, rel_embed: ExternalEmbedding):
        rel_ids = batch_g.edata['tid']
        ent_ids = batch_g.ndata['nid']
        ent_features = ent_embed(ent_ids)
        rel_features = rel_embed(rel_ids)
        # print(ent_features.shape, rel_features.shape)
        with batch_g.local_scope():
            h = ent_features
            for l in range(self.num_layers):
                h = self.gat_layers[l](batch_g, h, rel_features)
            batch_g.ndata['h'] = h

            unbatched_graphs = dgl.unbatch(batch_g)
            graph_cls_embed = torch.stack([sub_graph.dstdata['h'][0] for sub_graph in unbatched_graphs], dim=0)
            return graph_cls_embed

class ContrastiveKEModel(nn.Module):
    def __init__(self, args):
        super(ContrastiveKEModel, self).__init__()
        self.args = args
        self.entity_emb = ExternalEmbedding(num=args.n_entities, dim=args.ent_dim)
        self.relation_emb = ExternalEmbedding(num=args.n_relations, dim=args.rel_dim)

        self.n_entities = args.n_entities
        self.n_relations = args.n_relations
        self.kg_ent_dim = args.ent_dim
        self.kg_rel_dim = args.rel_dim
        self.graph_hidden_dim = args.graph_hidden_dim
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if args.graph_hidden_dim != args.rel_dim:
            self.rel_map = nn.Linear(args.rel_dim, args.graph_hidden_dim, bias=False)
        else:
            self.rel_map = Identity()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.eps = EMB_INIT_EPS
        self.ent_emb_init = (args.gamma + self.eps) / args.ent_dim
        self.rel_emb_init = (args.gamma + self.eps) / args.rel_dim
        self.graph_encoder = KGEGraphEncoder(num_layers=args.layers, in_ent_dim=args.ent_dim, in_rel_dim=args.rel_dim,
                                           hidden_dim=args.graph_hidden_dim, head_num=args.head_num,
                                           feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                                           activation=None, negative_slope=args.negative_slope,
                                           residual=args.residual, diff_head_tail=args.diff_head_tail)
        self.initialize_parameters()
        self.xent_loss = GraphContrastiveLoss(temperature=args.xent_temperature)


    def initialize_parameters(self):
        """Re-initialize the model.
        """
        self.entity_emb.init(self.ent_emb_init)
        self.relation_emb.init(self.rel_emb_init)
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.rel_map, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_ent.weight, gain=gain)

    def forward(self, g):
        self.relation_emb = self.rel_map.forward(self.relation_emb)
        with g.local_scope():
            cls_embed = self.graph_encoder.forward(batch_g=g, ent_embed=self.entity_emb, rel_embed=self.relation_emb)
            return cls_embed

    def loss_computation(self, cls_embed):
        loss = self.xent_loss(cls_embed)
        return loss

    def relation_embed(self):
        self.relation_emb = self.rel_map.forward(self.relation_emb)
        return self.relation_emb

class GraphContrastiveLoss(nn.Module):
    def __init__(self, temperature: float=0.1):
        super(GraphContrastiveLoss, self).__init__()
        self.xent_loss = NTXentLoss(temperature=temperature)

    def forward(self, cls_embed: T):
        batch_size = cls_embed.shape[0] // 2
        labels = torch.arange(0, batch_size, device=cls_embed.device).\
            repeat(2).view(2, batch_size).transpose(0,1).flatten()
        loss = self.xent_loss(cls_embed, labels)
        return loss