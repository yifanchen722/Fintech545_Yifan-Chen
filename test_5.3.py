import numpy as np
import pandas as pd
import test_lib

cov_matrix = np.loadtxt("../data/test5_3.csv", delimiter=",", skiprows=1)

Y = test_lib.get_near_psd(cov_matrix, 1e-10)

K = 100000
L = test_lib.chol_psd(Y, epsilon=1e-8)
M = np.random.normal(loc=0, scale=1, size=(Y.shape[0], K))

X = (L @ M).T
sample_cov = np.cov(X, rowvar=False)

print("Sample Covariance Matrix:\n", sample_cov)
