import numpy as np
import pandas as pd
import test_lib as tl

file_path = "../data/test5_2.csv"
Sigma = pd.read_csv(file_path)

K = 100000
L = tl.chol_psd(Sigma.values)
M = np.random.normal(loc=0, scale=1, size=(Sigma.shape[0], K))

X = (L @ M).T
sample_cov = np.cov(X, rowvar=False)
print("Sample Covariance Matrix:\n", sample_cov)
