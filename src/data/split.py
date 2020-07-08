import math
import random
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

def split_data(data: List[str], weights: Tuple = (0.8, 0.2, 0.0), seed: int = 100) -> Dict:
    split = {}

    data_new = data

    random.seed(seed)
    random.shuffle(data_new)

    total_words = len(data_new)
    train_limit = math.floor(total_words * weights[0])
    test_limit = math.floor(total_words * weights[1] + total_words * weights[0])

    split['train'] = data_new[:train_limit]
    split['test'] = data_new[train_limit:test_limit]
    split['validation'] = data_new[test_limit:]

    return split
