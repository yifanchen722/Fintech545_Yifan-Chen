import numpy as np
import pandas as pd

with open('testout_1.3.csv', 'r') as f:
    lines = f.readlines()

lines = lines[1:]
data = [list(map(float, line.strip().split(','))) for line in lines]
cov_matrix = np.array(data)

def near_psd(A, epsilon=1e-8):
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals[eigvals < epsilon] = epsilon
    psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    psd_matrix = (psd_matrix + psd_matrix.T) / 2
    return psd_matrix

near_cov = near_psd(cov_matrix)
near_cov_df = pd.DataFrame(near_cov, columns=['x1','x2','x3','x4','x5'])
near_cov_df.to_csv('testout_1.3_nearPSD.csv', float_format='%.15f', index=False)
print(near_cov_df)
