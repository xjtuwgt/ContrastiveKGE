from torch import nn
from kgeutils.scoreutils import TransEScore, DistMultScore, SimplEScore
from dglke.models.kgembedder import ExternalEmbedding
EMB_INIT_EPS = 2.0

class KEModel(nn.Module):
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma):
        super(KEModel, self).__init__()
        self.args = args
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = EMB_INIT_EPS
        ######################################################
        entity_dim = hidden_dim
        relation_dim = hidden_dim

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
