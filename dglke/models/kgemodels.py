from torch import nn
from kgeutils.scorefunction import TransEScore, DistMultScore, SimplEScore
from dglke.models.kgembedder import ExternalEmbedding
EMB_INIT_EPS = 2.0

class KEModel(nn.Module):
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma, hop_num,
                 double_entity_emb=False, double_relation_emb=False, add_special=False,
                 special_entity_dict=None, special_relation_dict=None):
        super(KEModel, self).__init__()
        self.args = args
        self.has_edge_importance = args.has_edge_importance
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hop_num = hop_num
        self.hidden_dim = hidden_dim
        self.eps = EMB_INIT_EPS
        ######################################################
        self.add_special_ = add_special
        self.special_entity_dict = special_entity_dict
        self.special_relation_dict = special_relation_dict
        if self.add_special_:
            assert special_relation_dict is not None and special_entity_dict is not None
            self.n_entities = self.n_entities + len(self.special_entity_dict)
            self.n_relations = self.n_relations + len(self.special_relation_dict)
        ######################################################

        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        self.entity_emb = ExternalEmbedding(num=self.n_entities, dim=entity_dim)
        self.relation_emb = ExternalEmbedding(num=self.n_relations, dim=relation_dim)

        self.rel_dim = relation_dim
        self.entity_dim = entity_dim
        self.emb_init = (gamma + self.eps) / hidden_dim

        if model_name == 'TransE' or model_name == 'TransE_l2':
            self.score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            self.score_func = TransEScore(gamma, 'l1')
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'SimplE':
            self.score_func = SimplEScore()
        else:
            ValueError('Score function {} not supported'.format(model_name))


    def initialize_parameters(self):
        """Re-initialize the model.
        """
        self.entity_emb.init(self.emb_init)
        self.score_func.reset_parameters()

    def save_emb(self, path, dataset):
        """Save the model.
        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.save(path, dataset+'_'+self.model_name+'_entity')
        if self.strict_rel_part or self.soft_rel_part:
            self.global_relation_emb.save(path, dataset+'_'+self.model_name+'_relation')
        else:
            self.relation_emb.save(path, dataset+'_'+self.model_name+'_relation')

        self.score_func.save(path, dataset+'_'+self.model_name)

    def load_emb(self, path, dataset):
        """Load the model.
        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset+'_'+self.model_name)
