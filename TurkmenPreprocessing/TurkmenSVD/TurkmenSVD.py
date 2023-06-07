"""

SVD

"""

import numpy as np
from scipy.sparse.linalg import svds

K = 1024

matrix_doc_word = np.load("MATRIX_DOC_WORD.npy")
u, sigma, vt = svds(matrix_doc_word.T, K)

np.save("U", u)
np.save("Sigma", sigma)
np.save("VT", vt)
