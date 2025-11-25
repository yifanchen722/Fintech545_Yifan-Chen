import test_lib as tl
import numpy as np

path = "../data/problem4.csv"

lam1 = 0.97
lam2 = 0.94

cov1 = tl.get_ew_cov(path, lam1, has_date=False, is_rate=True)

corr = tl.get_ew_corr(path, lam2, has_date=False, is_rate=True)


cov2 = np.zeros((cov1.shape[0], cov1.shape[1]))

for i in range(0, cov1.shape[0]):
    for j in range(0, cov1.shape[1]):
        cov2[i, j] = corr[i, j] * np.sqrt(cov1[i, i]) * np.sqrt(cov1[j, j])

print(cov2)
