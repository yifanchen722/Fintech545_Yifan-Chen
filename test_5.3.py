import numpy as np
import pandas as pd
import test_lib

cov_matrix = np.loadtxt(
    "/Users/nico/Desktop/input/test5_3.csv", delimiter=",", skiprows=1
)

Sigma = test_lib.get_near_psd(cov_matrix, 1e-10)

K = 100000
L = np.linalg.cholesky(Sigma)
M = np.random.normal(loc=0, scale=1, size=(Sigma.shape[0], K))

X = (L @ M).T
sample_cov = np.cov(X, rowvar=False)

np.set_printoptions(precision=15, suppress=True)

output_path = "/Users/nico/Desktop/output_5.3.csv"
pd.DataFrame(sample_cov).to_csv(output_path, index=False, header=False)
