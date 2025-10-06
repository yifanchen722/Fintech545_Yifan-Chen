import numpy as np
import test_lib
import pandas as pd

C = np.loadtxt("/Users/nico/Desktop/test5_3.csv", delimiter=",", skiprows=1)
var = np.diag(np.diag(C))
C = np.linalg.inv(np.sqrt(var)) @ C @ np.linalg.inv(np.sqrt(var))

maxiter = 100000
tol = 1e-10

deltaS = np.zeros((C.shape[0], C.shape[1]))
Y = C
gamma_1 = np.finfo(np.float64).max
gamma_2 = np.finfo(np.float64).max
R = np.zeros((C.shape[0], C.shape[1]))
X = np.zeros((C.shape[0], C.shape[1]))


for k in range(1, maxiter + 1):
    gamma_1 = gamma_2
    R = Y - deltaS
    X = test_lib.PSA(R)
    deltaS = X - R
    Y = test_lib.PUA(X)
    gamma_2 = test_lib.Frobeniusnorm(Y - C)
    print("iter:", k, "gamma_2:", gamma_2)
    if abs(gamma_1 - gamma_2) <= tol:
        break

Y = np.sqrt(var) @ Y @ np.sqrt(var)
print(Y)

K = 100000
L = np.linalg.cholesky(Y)
M = np.random.normal(loc=0, scale=1, size=(Y.shape[0], K))

V = (L @ M).T
sample_cov = np.cov(V, rowvar=False)

np.set_printoptions(precision=15, suppress=True)

output_path = "/Users/nico/Desktop/output_5.4.csv"
pd.DataFrame(sample_cov).to_csv(output_path, index=False, header=False)
