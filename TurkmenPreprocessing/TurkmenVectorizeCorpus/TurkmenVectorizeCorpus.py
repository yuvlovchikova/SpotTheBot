"""

Vectorize Corpus

"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

VECTOR_SHAPE = 8


def _cut_dict_values(input_dict: dict, k: int) -> dict:
    lambda_ = lambda value: value[:k]
    dict_ = dict(zip(input_dict, map(lambda_, input_dict.values())))
    return dict_

def vectorize_text(text: str, word_vector_dict: dict, vector_shape: int) -> dict:
    word_vector_dict = _cut_dict_values(word_vector_dict, vector_shape)
    words = text.split()
    vectors = list(map(word_vector_dict.get, words))
    text_dict = dict(zip(words, vectors))
    return text_dict

def vectorize_corpus(corpus_array: list,
                     pg_indexes: list,
                     word_vector_dict: dict,
                     vector_shape: int) -> np.ndarray:
    result = list()
    current_page_index = -1
    sentence_num = 0
    for pg_index, sentence in tqdm(zip(pg_indexes, corpus_array), total=len(pg_indexes)):
        vectorized_sentence = vectorize_text(sentence, word_vector_dict,
                                             vector_shape)
        if current_page_index != pg_index:
            current_page_index = pg_index
            sentence_num = 0
        result.append({'article_index': pg_index,
                       'sentence_num': sentence_num,
                       'sentence': vectorized_sentence})
        sentence_num += 1
    return np.array(result, dtype=object)


corpus = pd.read_csv("TurkmenCleanCorpusDotSplit.csv")
word_list = np.load("WORD_LIST.npy")
U = np.load("U.npy")
Sigma = np.load("Sigma.npy")

words_vectors = np.dot(U, np.diag(Sigma))

word_vector_dict = dict(zip(word_list, words_vectors))
vectorized_corpus = vectorize_corpus(corpus['clean_text'].tolist(),
                                     corpus['article_index'].tolist(),
                                     word_vector_dict,
                                     VECTOR_SHAPE)


np.save("TurkmenVectorizedCorpus", vectorized_corpus)

lambda_ = lambda vector: vector.tolist()
dict_ = dict(zip(word_vector_dict, map(lambda_, word_vector_dict.values())))
with open('WordVectorDict.json', 'w') as f:
    json.dump(dict_, f)
