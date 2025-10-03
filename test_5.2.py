import numpy as np
import pandas as pd

file_path = "/Users/nico/Desktop/input/test5_2.csv"
Sigma = pd.read_csv(file_path)

K = 100000
L = np.linalg.cholesky(Sigma)
M = np.random.normal(loc=0, scale=1, size=(Sigma.shape[0], K))

X = (L @ M).T
sample_cov = np.cov(X, rowvar=False)

np.set_printoptions(precision=15, suppress=True)

output_path = "/Users/nico/Desktop/output_5.2.csv"
pd.DataFrame(sample_cov).to_csv(output_path, index=False, header=False)
