"""

Get Words Vectors Dict

"""

import sys
import json
import numpy as np
import pandas as pd


VECTOR_SHAPE = int(sys.argv[1])


def _process_dict_values(input_dict: dict, k: int) -> dict:
    dict_ = {}
    for key, value in input_dict.items():
        dict_[key] = value.tolist()[:k]
    return dict_
    
def get_word_vec_dict(U: np.ndarray,
                      Sigma: np.ndarray,
                      word_list: np.ndarray,
                      vector_shape: int) -> dict:
    words_vectors = np.dot(U, np.diag(Sigma))
    word_vector_dict = dict(zip(word_list, words_vectors))
    word_vector_dict = _process_dict_values(word_vector_dict, vector_shape)
    return word_vector_dict
    

word_list = np.load(sys.argv[2])
U = np.load(sys.argv[3])
Sigma = np.load(sys.argv[4])

word_vector_dict = get_word_vec_dict(U, Sigma, word_list, VECTOR_SHAPE)


save_name = "RuWordVectorDict" + str(VECTOR_SHAPE) + ".json"
with open(save_name, 'w') as f:
    json.dump(word_vector_dict, f)
