import numpy as np
import test_lib

corr_matrix = np.loadtxt("../data/testout_1.4.csv", delimiter=",", skiprows=1)

C_hat = test_lib.get_near_psd(corr_matrix, 1e-6)

print(C_hat)
