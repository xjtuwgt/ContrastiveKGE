import random
from typing import Optional
import os

import numpy as np
import torch
import dgl

def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    return seed