from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class KGETrainDataset(Dataset):
    def __init__(self, dataset,  negative_sample_size, add_special=False, reverse=False, has_importance=False):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)
        self.n_entities = dataset.n_entities
        self.n_relations = dataset.n_relations
        self.reverse = reverse
        self.add_special = add_special
        self.has_importance = has_importance
        self.negative_sample_size = negative_sample_size
        self.len = len(triples)
