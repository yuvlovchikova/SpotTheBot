"""

TF-IDF

"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TOKEN_PATTERN = r"\S+"


def create_doc_word_matrix(corpus: pd.DataFrame, token_pattern: str) -> tuple:
    vectorizer = TfidfVectorizer(token_pattern=token_pattern)
    matrix_doc_word = vectorizer.fit_transform(corpus['clean_text'].values)
    return matrix_doc_word.toarray(), np.array(vectorizer.get_feature_names())

corpus = pd.read_csv("TurkmenCleanCorpus.csv")
matrix_doc_word, word_list = create_doc_word_matrix(corpus, TOKEN_PATTERN)
np.save("MATRIX_DOC_WORD", matrix_doc_word)
np.save("WORD_LIST", word_list)
