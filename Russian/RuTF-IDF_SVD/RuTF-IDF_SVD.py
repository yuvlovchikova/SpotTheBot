"""

TF-IDF & SVD

"""


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#TF-IDF

TOKEN_PATTERN = r"([А-Яа-я]{1,})"

def create_doc_word_matrix(corpus: pd.DataFrame,
                           token_pattern: str,
                           text_column: str) -> tuple:
    vectorizer = TfidfVectorizer(token_pattern=token_pattern)
    matrix_doc_word = vectorizer.fit_transform(corpus[text_column].values)
    return matrix_doc_word.toarray(), np.array(vectorizer.get_feature_names())


corpus = pd.read_csv("RuPreprocessedNoSep.csv")
matrix_doc_word, word_list = create_doc_word_matrix(corpus, TOKEN_PATTERN, 'preprocessed_text_no_sep')

np.save("MATRIX_DOC_WORD", matrix_doc_word)
np.save("WORD_LIST", word_list)

#SVD

u, sigma, vt = np.linalg.svd(matrix_doc_word.T, full_matrices=False)

np.save("U", u)
np.save("Sigma", sigma)
np.save("VT", vt)
