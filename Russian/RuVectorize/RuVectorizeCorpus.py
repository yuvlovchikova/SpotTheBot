"""

Vectorize Corpus

"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def vectorize_text(text: str, word_vector_dict: dict) -> dict:
    words = text.split()
    vectors = list(map(word_vector_dict.get, words))
    text_dict = dict(zip(words, vectors))
    return text_dict

def vectorize_corpus_with_dots(corpus: list,
                               separator: str,
                               word_vector_dict: dict) -> np.ndarray:
    vectorized_corpus = list()
    for text_index, text in enumerate(tqdm(corpus)):
        sentences = text.split(sep=separator)
        vectorized_text = list()
        for sentence_index, sentence in enumerate(sentences):
            if not sentence:
                continue
            vectorized_sentence = vectorize_text(sentence, word_vector_dict)
            vectorized_text.append({'document_index': text_index,
                                    'sentence_index': sentence_index,
                                    'sentence_text': vectorized_sentence})
        vectorized_corpus.append(vectorized_text)
    return np.array(vectorized_corpus, dtype=object)


corpus = pd.read_csv("/home/yuvlovchikova/Yulia/Russian/RuPreprocessed/RuPreprocessedWithSep.csv")
word_vector_dict = json.load(open("/home/yuvlovchikova/Yulia/Russian/RuPreprocessed/RuWordVectorDict8.json"))

vectorized_corpus = vectorize_corpus_with_dots(corpus['preprocessed_text'].tolist(),
                                               ' . ',
                                               word_vector_dict)

np.save("RuVectorizedCorpus8", vectorized_corpus)
