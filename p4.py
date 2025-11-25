import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import norm
import test_lib

path = "../data/problem4.csv"
lam = 0.97

cov = test_lib.get_ew_cov(path, lam, has_date=True, is_rate=False)
print(cov)

path = "../data/problem4.csv"
lam = 0.94

corr = test_lib.get_ew_corr(path, lam, has_date=False, is_rate=True)
print(corr)

path = "../data/problem4.csv"

lam1 = 0.97
lam2 = 0.94

cov1 = test_lib.get_ew_cov(path, lam1, has_date=False, is_rate=True)

corr = test_lib.get_ew_corr(path, lam2, has_date=False, is_rate=True)


cov2 = np.zeros((cov1.shape[0], cov1.shape[1]))

for i in range(0, cov1.shape[0]):
    for j in range(0, cov1.shape[1]):
        cov2[i, j] = corr[i, j] * np.sqrt(cov1[i, i]) * np.sqrt(cov1[j, j])

print(cov2)
