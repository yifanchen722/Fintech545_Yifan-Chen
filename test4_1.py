import numpy as np
import pandas as pd
import test_lib

# 判断是否为psd
data_path = "../data/testout_3.1.csv"
is_psd = test_lib.tell_psd(data_path)
print(is_psd)


def chol_psd(A, epsilon=1e-8):
    A = (A + A.T) / 2
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals[eigvals < epsilon] = epsilon
        A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        A_psd = (A_psd + A_psd.T) / 2
        L = np.linalg.cholesky(A_psd)
    return L


df = pd.read_csv("testout_3.1.csv", sep=",")
data = df.values.astype(float)

L = chol_psd(data)
df_L = pd.DataFrame(L, columns=df.columns, index=df.columns)
pd.options.display.float_format = "{:.16f}".format
print(df_L)
df_L.to_csv("chol_psd_testout_3.1.csv", sep=",")
