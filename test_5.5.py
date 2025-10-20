import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import test_lib

cov_matrix = np.loadtxt("../data/test5_2.csv", delimiter=",", skiprows=1)


eigenval, eigenvec = np.linalg.eig(cov_matrix)

eigenindex = np.argsort(eigenval)[::-1]

eigenval_sort = eigenval[eigenindex]
eigenvec_sort = eigenvec[:, eigenindex]

PCT = 0

for i in range(0, len(eigenval_sort)):
    if np.sum(eigenval_sort[0 : i + 1]) / np.sum(eigenval_sort) > 0.99:
        PCT = i + 1
        break
print(PCT)

S = eigenvec_sort[:, :PCT]
lam = sqrtm(np.diag(eigenval_sort[:PCT]))
L = B = S @ lam

K = 100000
M = np.random.normal(loc=0, scale=1, size=(L.shape[1], K))

X = (L @ M).T
sample_cov = np.cov(X, rowvar=False)

print("Sample Covariance Matrix:\n", sample_cov)
