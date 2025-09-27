import numpy as np
import pandas as pd

df = pd.read_csv("testout_1.4.csv", sep=",")
corr_matrix = df.values.astype(float)

def near_psd(A, epsilon=1e-8):
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals[eigvals < epsilon] = epsilon
    A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    A_psd = (A_psd + A_psd.T) / 2
    D_inv = np.diag(1 / np.sqrt(np.diag(A_psd)))
    A_psd = D_inv @ A_psd @ D_inv
    return A_psd

near_corr = near_psd(corr_matrix)
near_corr_df = pd.DataFrame(near_corr, columns=df.columns, index=df.columns)
pd.options.display.float_format = '{:.16f}'.format
print(near_corr_df)
near_corr_df.to_csv("near_psd_testout_1.4.csv", sep=",", index=True)

