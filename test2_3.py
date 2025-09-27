import re, sys, numpy as np, pandas as pd

path = "test2.csv"

def read_data(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    header = lines[0]
    cols = re.findall(r'[A-Za-z_]\w*', header)
    if not cols:
        cols = header.strip().split()
    nums = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?', "\n".join(lines[1:]))
    arr = np.array(nums, dtype=float)
    ncols = len(cols)
    if ncols == 0:
        sys.exit("无法识别列名")
    if arr.size % ncols != 0:
        arr = arr[:arr.size - arr.size % ncols]
    data = arr.reshape(-1, ncols)
    return pd.DataFrame(data, columns=cols)

def ew_cov(X, lam):
    X = X - X.mean(axis=0)
    n, d = X.shape
    powvec = lam ** (n - 1 - np.arange(n))
    weights = (1 - lam) * powvec / (1 - lam ** n)
    S = np.zeros((d, d))
    for t in range(n):
        xt = X[t].reshape(-1, 1)
        S += weights[t] * (xt @ xt.T)
    return S

def ew_corr(X, lam):
    S = ew_cov(X, lam)
    stddev = np.sqrt(np.diag(S))
    return S / np.outer(stddev, stddev)

df = read_data(path)
X = df.values

S_var = ew_cov(X, 0.97)
R_corr = ew_corr(X, 0.94)

d = S_var.shape[0]
S_final = np.zeros((d, d))
for i in range(d):
    for j in range(d):
        if i == j:
            S_final[i, j] = S_var[i, i]
        else:
            S_final[i, j] = R_corr[i, j] * np.sqrt(S_var[i, i] * S_var[j, j])

result = pd.DataFrame(S_final, index=df.columns, columns=df.columns)
print(result.to_string(float_format=lambda x: f"{x:.16f}"))
result.to_csv("ew_cov_with_var_corr.csv", sep="\t", float_format="%.16f")
