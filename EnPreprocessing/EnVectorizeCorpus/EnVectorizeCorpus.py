"""

Vectorize Corpus

"""

from tqdm import tqdm
import numpy as np
import pandas as pd

CORPUS = "EnCleanCorpus.csv"
WORD_LIST = "WORD_LIST.npy"
WORD_VECTORS = "U.npy"

VECTOR_SHAPE = 8
SEQUENCE_SHAPE = 2


def _cut_dict_values(input_dict: dict, k: int) -> dict:
    lambda_ = lambda value: value[:k]
    dict_ = dict(zip(input_dict, map(lambda_, input_dict.values())))
    return dict_

def _concat_columns(dataframe: pd.DataFrame) -> np.ndarray:
    if dataframe.shape[0] == 0:
        return np.array([], dtype=object)
    array = dataframe.iloc[0].values
    result = [array]
    for i in range(1, dataframe.shape[0]):
        curr_array = dataframe.iloc[i].values
        result = np.append(result, [curr_array], axis=0)
    return result

def create_ngrams(sentence: list, sequence_shape: int) -> pd.Series:
    if sequence_shape > len(sentence):
        return pd.Series(dtype='object')
    n_words = len(sentence) - sequence_shape + 1
    n_sentence = sentence[:n_words]
    ngrams = pd.Series(dtype='object')
    for i in range(len(n_sentence)):
        ngrams = ngrams.append(pd.Series([sentence[i:i + sequence_shape]]))
    ngrams.reset_index(drop=True, inplace=True)
    return ngrams

def vectorize_text(text: str,
                   word_vector_dict: dict,
                   vector_shape: int,
                   sequence_shape: int) -> np.ndarray:
    word_vector_dict = _cut_dict_values(word_vector_dict, vector_shape)
    sentence = text.split()
    ngrams = create_ngrams(sentence, sequence_shape)
    dataframe_ngrams = ngrams.apply(lambda ngram: pd.Series(ngram))
    dataframe_vectors = dataframe_ngrams.apply(lambda ngram: ngram.map(word_vector_dict))
    result_dataframe = _concat_columns(dataframe_vectors)
    return result_dataframe

def vectorize_corpus(corpus_array: np.ndarray,
                     word_vector_dict: dict,
                     vector_shape: int,
                     sequence_shape: int) -> list:
    result = list()
    for text_index in tqdm(range(len(corpus_array))):
        vectorized_text = vectorize_text(corpus_array[text_index][0], word_vector_dict,
                                         vector_shape, sequence_shape)
        result.append(vectorized_text)
    return result

def vectorize(corpus: str,
              word_list: str,
              word_vectors: str,
              vector_shape: int,
              sequence_shape: int) -> np.ndarray:
    corpus = pd.read_csv(corpus)
    word_list = np.load(word_list)
    word_vectors = np.load(word_vectors)
    word_vector_dict = dict(zip(word_list, word_vectors))
    vectorized_corpus = vectorize_corpus(corpus.values,
                                         word_vector_dict,
                                         vector_shape,
                                         sequence_shape)
    return np.array(vectorized_corpus, dtype=object)


vectorized_corpus = vectorize(CORPUS,
                              WORD_LIST,
                              WORD_VECTORS,
                              VECTOR_SHAPE,
                              SEQUENCE_SHAPE)
np.save("EnVectorizedCorpus", vectorized_corpus)