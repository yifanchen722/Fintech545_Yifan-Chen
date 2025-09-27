import numpy as np
import pandas as pd

def higham_corr(A, tol=1e-10, max_iter=100):
    n = A.shape[0]
    W = np.identity(n)
    Y = A.copy()
    for _ in range(max_iter):
        R = Y - W
        X = project_psd(R)
        W = X - R
        Y = project_unit_diag(X)
        if np.linalg.norm(Y - A, ord='fro') < tol:
            break
    return Y

def project_psd(A):
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < 0] = 0
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def project_unit_diag(A):
    B = A.copy()
    np.fill_diagonal(B, 1.0)
    return B

df = pd.read_csv("testout_1.4.csv")  
data = df.values.astype(float)

higham_result = higham_corr(data)
df_result = pd.DataFrame(higham_result, columns=df.columns, index=df.columns)
pd.options.display.float_format = '{:.16f}'.format
print(df_result)
df_result.to_csv("higham_near_psd.csv", sep="\t")
