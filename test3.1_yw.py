import numpy as np
import pandas as pd
import test_lib

# with open("testout_1.3.csv", "r") as f:
#     lines = f.readlines()

# lines = lines[1:]
# data = [list(map(float, line.strip().split(","))) for line in lines]
# cov_matrix = np.array(data)

cov_matrix = np.loadtxt("testout_1.3.csv", delimiter=",", skiprows=1)

C_hat = test_lib.get_near_psd(cov_matrix, 1e-6)

print(C_hat)

# eigenval, eigenvec = np.linalg.eig(cov_matrix)

# print(eigenval)
# print(eigenvec[0, 0])
