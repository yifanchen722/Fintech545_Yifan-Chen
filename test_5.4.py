import numpy as np
import test_lib
import pandas as pd

data_path = "../data/test5_3.csv"

maxiter = 100000
tol = 1e-10

Y = test_lib.higham(data_path, maxiter, tol)
print(Y)

K = 100000
L = np.linalg.cholesky(Y)
M = np.random.normal(loc=0, scale=1, size=(Y.shape[0], K))

V = (L @ M).T
sample_cov = np.cov(V, rowvar=False)

np.set_printoptions(precision=15, suppress=True)

output_path = "/Users/nico/Desktop/output_5.4.csv"
pd.DataFrame(sample_cov).to_csv(output_path, index=False, header=False)
