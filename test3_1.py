import numpy as np
import pandas as pd
import test_lib

cov_matrix = np.loadtxt("../data/testout_1.3.csv", delimiter=",", skiprows=1)

C_hat = test_lib.get_near_psd(cov_matrix, 1e-6)

print(C_hat)

# eigenval, eigenvec = np.linalg.eig(cov_matrix)

# print(eigenval)
# print(eigenvec[0, 0])
