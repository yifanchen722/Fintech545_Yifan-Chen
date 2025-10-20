import numpy as np
import test_lib
import pandas as pd

data_path = "../data/test5_3.csv"

maxiter = 100000
tol = 1e-10

Y = test_lib.higham(data_path, maxiter, tol)
print(Y)

K = 100000
L = test_lib.chol_psd(Y, epsilon=1e-8)
M = np.random.normal(loc=0, scale=1, size=(Y.shape[0], K))

V = (L @ M).T
sample_cov = np.cov(V, rowvar=False)

print("Sample Covariance Matrix from Simulated Data:\n", sample_cov)
