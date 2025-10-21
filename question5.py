import numpy as np
import pandas as pd
import test_lib as tl

df = pd.read_csv("../data/problem5.csv")
pairwise_cov = df.cov()

print(pairwise_cov)
pairwise_cov.to_csv("../output/pairwise_cov.csv", index=False)

data_path = "../output/pairwise_cov.csv"
is_psd = tl.tell_psd(data_path)
print(is_psd)

data_path = "../output/pairwise_cov.csv"

maxiter = 100000
tol = 1e-10

Y = tl.higham(data_path, maxiter, tol)
print(Y)
