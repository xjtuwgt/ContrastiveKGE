from torch import nn
from kgeutils.scoreutils import TransEScore, DistMultScore, SimplEScore
from dglke.models.kgembedder import ExternalEmbedding
EMB_INIT_EPS = 2.0

class KGEModel(nn.Module):
    def __init__(self, args):
        super(KGEModel, self).__init__()
        self.args = args
        self.n_entities = args.n_entities
        self.n_relations = args.n_relations
        self.model_name = args.model_name
        self.hidden_dim = args.hidden_dim
        self.eps = EMB_INIT_EPS
        ######################################################
        entity_dim = self.hidden_dim
        relation_dim = self.hidden_dim

        self.entity_emb = ExternalEmbedding(num=self.n_entities, dim=entity_dim)
        self.relation_emb = ExternalEmbedding(num=self.n_relations, dim=relation_dim)

        self.rel_dim = relation_dim
        self.entity_dim = entity_dim
        self.emb_init = (args.gamma + self.eps) / self.hidden_dim

        if args.model_name == 'TransE' or args.model_name == 'TransE_l2':
            self.score_func = TransEScore(args.gamma, 'l2')
        elif args.model_name == 'TransE_l1':
            self.score_func = TransEScore(args.gamma, 'l1')
        elif args.model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif args.model_name == 'SimplE':
            self.score_func = SimplEScore()
        else:
            ValueError('Score function {} not supported'.format(args.model_name))


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
