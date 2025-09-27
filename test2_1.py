import re, sys, numpy as np, pandas as pd

path = "test2.csv"
lam = 0.97

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
    extra = arr.size % ncols
    arr = arr[:arr.size - extra]
data = arr.reshape(-1, ncols)
df = pd.DataFrame(data, columns=cols)

X = df.values
X = X - X.mean(axis=0)
n, d = X.shape
powvec = lam ** (n - 1 - np.arange(n))
weights = (1 - lam) * powvec / (1 - lam ** n)

S = np.zeros((d, d))
for t in range(n):
    xt = X[t].reshape(-1, 1)
    S += weights[t] * (xt @ xt.T)

cov_df = pd.DataFrame(S, index=cols, columns=cols)
print(cov_df.to_string(float_format=lambda x: f"{x:.16f}"))
cov_df.to_csv("ew_covariance.csv", sep="\t", float_format="%.16f")
