import test_lib as tl

data_path = "../data/problem5.csv"

maxiter = 100000
tol = 1e-10

Y = tl.higham(data_path, maxiter, tol)
print(Y)
