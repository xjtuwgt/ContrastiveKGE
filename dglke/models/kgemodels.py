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
        self.hidden_dim = args.graph_hidden_dim
        self.eps = EMB_INIT_EPS
        ######################################################
        entity_dim = self.hidden_dim
        relation_dim = self.hidden_dim

        self.entity_embedding = ExternalEmbedding(num=self.n_entities, dim=entity_dim)
        self.relation_embedding = ExternalEmbedding(num=self.n_relations, dim=relation_dim)

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
        self.entity_embedding.init(self.emb_init)
        self.relation_embedding.init(self.emb_init)

    def initialize_parameters_with_emb(self, path):

        return

    def save_emb(self, path, dataset):
        """Save the model.
        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_embedding.save(path, dataset+'_'+self.model_name+'_entity')
        self.relation_embedding.save(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.save(path, dataset+'_'+self.model_name+'_score')

    def load_emb(self, path, dataset):
        """Load the model.
        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_embedding.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_embedding.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset+'_'+self.model_name+'_score')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        if mode == 'single':
            head = self.entity_embedding(sample[:,0]).unsqueeze(1)
            relation = self.relation_embedding(sample[:,1]).unsqueeze(1)
            tail = self.entity_embedding(sample[:,2]).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = self.entity_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = self.relation_embedding(tail_part[:,1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part[:,2]).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = self.entity_embedding(head_part[:,0]).unsqueeze(1)
            relation = self.relation_embedding(head_part[:,1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        score = self.score_func.forward(head=head, tail=tail, relation=relation, mode=mode)['score']
        return score