import numpy as np
import pandas as pd

with open('testout_1.3.csv', 'r') as f:
    lines = f.readlines()

lines = lines[1:]
data = [list(map(float, line.strip().split(','))) for line in lines]
A = np.array(data)

def higham_near_psd(A, tol=1e-8, max_iter=100):
    n = A.shape[0]
    X = A.copy()
    Y = np.zeros_like(A)
    for _ in range(max_iter):
        R = X - Y
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals[eigvals < 0] = 0
        X_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Y = X_new - R
        if np.linalg.norm(X_new - X, ord='fro') < tol:
            break
        X = X_new
    return (X + X.T) / 2

near_cov = higham_near_psd(A)
near_cov_df = pd.DataFrame(near_cov, columns=['x1','x2','x3','x4','x5'])
near_cov_df.to_csv('testout_1.3_Higham.csv', float_format='%.15f', index=False)
print(near_cov_df)
