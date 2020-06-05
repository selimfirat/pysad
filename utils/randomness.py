import random
import numpy as np


def fix_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
