"""

SVD

"""

import numpy as np
from scipy.sparse.linalg import svds

K = 1024
MATRIX_WORD_DOC = "MATRIX_WORD_DOC.npy"
U = "U"
SIGMA = "Sigma"
VT = "VT"


def SVD(path: str, k: int) -> np.array:
    return svds(np.load(path).T, k)


u, sigma, vt = SVD(MATRIX_WORD_DOC, K)
np.save(U, u)
np.save(SIGMA, sigma)
np.save(VT, vt)
